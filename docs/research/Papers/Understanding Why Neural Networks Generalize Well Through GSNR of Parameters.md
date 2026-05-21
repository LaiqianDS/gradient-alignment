---
authors:
  - Jinlong Liu
  - Yunzhi Bai
  - Guoqing Jiang
  - Ting Chen
  - Huayan Wang
year: 2020
status: to-read
relevance: high
url: https://arxiv.org/abs/2001.07384
last_review: 2026-05-07
---

# Understanding Why Neural Networks Generalize Well Through GSNR of Parameters

## Summary

**Contextualización.** Las redes neuronales profundas (DNN) suelen contener muchos más parámetros entrenables que muestras de entrenamiento, configuración que, según las teorías clásicas de la generalización (dimensión VC, complejidad de Rademacher), debería producir un *gap* de generalización elevado, esto es, una diferencia grande entre la pérdida en entrenamiento y la pérdida en test. Sin embargo, en la práctica las DNN exhiben *gaps* de generalización notablemente pequeños. Trabajos previos han atribuido este fenómeno a propiedades de la optimización por descenso de gradiente (Hardt et al., 2015; Zhang et al., 2016) o a la curvatura del paisaje de pérdida (Keskar et al., 2016; Dinh et al., 2017), pero el mecanismo concreto que vincula la dinámica del entrenamiento con la capacidad de generalización seguía sin estar plenamente caracterizado. El paper de Liu et al. (ICLR 2020) aborda directamente esta cuestión proporcionando un puente cuantitativo entre la estadística de los gradientes durante el entrenamiento y el *gap* de generalización resultante.

**Aportación.** La contribución central es la introducción del **Gradient Signal-to-Noise Ratio (GSNR)** como métrica por parámetro que conecta de forma cuantitativa la dinámica del descenso de gradiente con la generalización de la red. El GSNR de un parámetro $\theta_j$ se define como el cociente entre el cuadrado de la media de su gradiente sobre la distribución de datos y la varianza de dicho gradiente sobre la misma distribución, es decir, $r(\theta_j) = \tilde{g}^2(\theta_j) / \rho^2(\theta_j)$, donde $\tilde{g}(\theta_j) = \mathbb{E}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$ y $\rho^2(\theta_j) = \mathrm{Var}_{(x,y)\sim\mathcal{Z}}[g(x,y,\theta_j)]$. Intuitivamente, un GSNR alto indica que la mayoría de las muestras de entrenamiento coinciden en la dirección óptima de actualización para ese parámetro, lo que sugiere que el parámetro está aprendiendo un patrón consistente y, por tanto, transferible. Los autores demuestran que mantener un GSNR elevado durante el entrenamiento implica un *gap* de generalización pequeño, y prueban analíticamente que la dinámica del descenso de gradiente en DNN —a diferencia de la de modelos lineales como regresión logística o SVM— produce naturalmente un GSNR elevado a través de un proceso vinculado al aprendizaje de características (*feature learning*).

**Metodología.** El núcleo teórico es la derivación del **One-Step Generalization Ratio (OSGR)**, definido como $R(\mathcal{Z}, n) = \mathbb{E}[\Delta L[D']]/\mathbb{E}[\Delta L[D]]$, es decir, la ratio entre la mejora esperada de la pérdida en test y la mejora esperada de la pérdida en entrenamiento tras un único paso de descenso de gradiente. Bajo la *non-overfitting limit approximation* (Asunción 2.3.1: $g_D(\theta)$ y $g_{D'}(\theta)$ siguen la misma distribución) y considerando el límite de tasa de aprendizaje pequeña ($\lambda \to 0$), los autores derivan una expresión cerrada que liga el OSGR con el GSNR de los parámetros: $R(\mathcal{Z}, n) = 1 - \frac{1}{n}\sum_j W_j \cdot \frac{1}{r_j + 1/n}$, donde $W_j$ es un peso proporcional a la contribución del parámetro $j$ a la disminución de la pérdida de entrenamiento y $\sum_j W_j = 1$. Esta expresión muestra de forma transparente que, cuando los $r_j$ son grandes, el OSGR tiende a 1 (el test loss decrece prácticamente al mismo ritmo que el train loss, *gap* pequeño), y cuando son pequeños, el OSGR cae (puede acercarse a 0 o incluso ser negativo, *gap* grande). El paso clave en la derivación consiste en escribir $g_D(\theta) = \tilde{g}(\theta) + \epsilon$ y $g_{D'}(\theta) = \tilde{g}(\theta) + \epsilon'$ con $\epsilon, \epsilon'$ ruidos independientes de media cero y varianza $\sigma^2(\theta) = \rho^2(\theta)/n$, y aplicar la aproximación lineal $\Delta L \approx -\Delta\theta \cdot \partial L/\partial\theta$. La sección 3 complementa esta derivación con un análisis de la dinámica de capas plenamente conectadas (ec. 27–34, Apéndice B), mostrando que el primer término del cambio en un paso de $g_D^{(l)}$ tiene signo opuesto a los pesos $W^{(l)}_{s,c}$, lo que genera un proceso de retroalimentación positiva: los pesos cuyo signo es contrario al de su gradiente medio tienden a aumentar su valor absoluto, incrementando el numerador del GSNR.

**Datasets y modelos.** Los experimentos se realizan sobre **MNIST** con una CNN simple de dos bloques Conv-ReLU-MaxPooling más dos capas plenamente conectadas (Tabla 2, $p \in \{6, 8, 10, 12, 14, 16, 18, 20\}$ canales), sobre **CIFAR-10** con una red más profunda con Batch Normalization (4 bloques Conv-BN-ReLU-Conv-BN-ReLU-MaxPooling más 3 FC, Tabla 3) y sobre **ResNet18** para comparar GSNR con etiquetas reales frente a etiquetas aleatorizadas. Adicionalmente se utiliza un *toy dataset* sintético generado por $y = x_0 x_1 + \epsilon$ con un MLP de dos capas (2 entradas, $N$ neuronas ocultas, 1 salida). Las configuraciones experimentales varían el tamaño del conjunto de entrenamiento ($n \in \{1000, 2000, 4000, 6000, 8000, 10000, 15000\}$ en MNIST), la probabilidad de ruido en las etiquetas ($p_{random} \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$) y la anchura de la red, todo entrenado con descenso de gradiente puro (no SGD) y tasa de aprendizaje pequeña ($\lambda = 0.001$).

**Métricas.** Las métricas reportadas incluyen: (i) los **valores de GSNR** estimados a lo largo del entrenamiento ($r_{j,t} \approx g_{D,j,t}^2 / \rho^2_{D,j,t}$, ec. 25); (ii) la **train loss** $L[D]$ y la **test loss** $L[D']$; (iii) el **OSGR** estimado empíricamente promediando sobre $M=10$ conjuntos de entrenamiento (ec. 23–24); (iv) la **correlación de Pearson** entre el lado izquierdo (OSGR por definición) y el lado derecho (OSGR como función del GSNR) de la ec. 19, que alcanza valores entre 0.907 y 0.968 a lo largo del entrenamiento en MNIST; (v) la proporción $p_{same\_sign}$ de parámetros cuyo gradiente tiene el mismo signo entre muestras distintas, que crece de aproximadamente 50% (inicialización aleatoria) hasta 56% durante el entrenamiento; (vi) la correlación entre los pesos $W^{(l)}_{s,c}$ y la variación $\Delta g^{(l)}_{D,s,c}$, que resulta negativa (especialmente acusada en el top-10% de pesos de mayor valor absoluto, donde ~80% tienen signo opuesto a su gradiente medio) y confirma así el proceso de retroalimentación positiva sobre $(g^{(l)}_{D,s,c})^2$; y (vii) el GSNR medio para configuraciones congeladas frente a no congeladas en el experimento del MLP de dos capas, demostrando que sin *feature learning* el GSNR decrece monótonamente.

**Conclusiones.** Los resultados establecen empírica y teóricamente que un **GSNR alto durante el entrenamiento es equivalente a un *gap* de generalización pequeño**: el OSGR se aproxima a 1 cuando la mayor parte de los parámetros mantiene gradientes consistentes entre muestras. Más aún, los autores muestran que la curva de GSNR en DNN entrenadas con datos reales presenta un **proceso ascendente característico en las primeras etapas del entrenamiento**, mientras que con etiquetas aleatorizadas el GSNR permanece bajo durante todo el entrenamiento; esta diferencia explica por qué redes con la misma capacidad de memorización pueden generalizar de forma radicalmente distinta. La causa subyacente es el **aprendizaje de características** (*feature learning*) propio de las DNN: a medida que las capas inferiores aprenden representaciones útiles, los gradientes de las capas superiores se vuelven más coherentes entre muestras, elevando el GSNR. En modelos lineales (regresión logística, SVM) este mecanismo no existe y el GSNR decrece monótonamente durante el entrenamiento, lo que coincide con sus mayores *gaps* de generalización. Los autores postulan que este vínculo entre dinámica de optimización y generalización es probablemente la clave del éxito empírico de las DNN.

## Medición y pipeline

**Métrica concreta.** El estadístico propuesto por Liu et al. (2020) es el *Gradient Signal-to-Noise Ratio* por parámetro, definido como $r(\theta_j) = \mathbb{E}_{(x,y)\sim\mathcal{Z}}[g_j(x,y,\theta)]^2 \,/\, \mathrm{Var}_{(x,y)\sim\mathcal{Z}}[g_j(x,y,\theta)]$, donde la esperanza y la varianza se calculan sobre la distribución empírica de muestras de entrenamiento. En la práctica del paper se estima sobre un subconjunto de $N$ ejemplos y se reporta tanto el GSNR por parámetro como agregaciones (media sobre todos los parámetros y media por capa).

**Entradas.** El estimador requiere gradientes *per-example* $g_{i,j}$ para cada ejemplo $i \in \{1,\dots,N\}$ y cada parámetro $j$, lo que excluye el gradiente promediado por minibatch que devuelve `backward()` por defecto. La implementación se apoya en `torch.func.vmap` (antes `functorch`) o, alternativamente, en `opacus.GradSampleModule`, que expone `param.grad_sample` por ejemplo.

**Cuándo computar.** Una estimación por época (o cada $k$ épocas en regímenes largos) sobre un subconjunto fijo de $N$ ejemplos del conjunto de entrenamiento. El coste por evaluación es elevado, por lo que conviene desacoplar la frecuencia del logging de la frecuencia de actualización de pesos.

**Coste.** El cálculo naïve requiere $N$ pasadas *forward-backward* de un ejemplo y, si se almacenan todos los gradientes, memoria $\mathcal{O}(N\cdot P)$ con $P$ el número de parámetros. La alternativa adoptada es un estimador *streaming*: acumular $\sum_i g_{i,j}$ y $\sum_i g_{i,j}^2$ por parámetro, obteniendo media y varianza online con memoria $\mathcal{O}(P)$ y tiempo $\mathcal{O}(N)$.

**Integración en el pipeline.** Pseudocódigo del cómputo por época:

```python
sum_g  = {p: torch.zeros_like(p) for p in model.parameters()}
sum_g2 = {p: torch.zeros_like(p) for p in model.parameters()}
for x_i, y_i in subset_loader:                  # N ejemplos
    loss_i = criterion(model(x_i), y_i)
    grads  = torch.autograd.grad(loss_i, model.parameters())
    for p, g in zip(model.parameters(), grads):
        sum_g[p]  += g.detach()
        sum_g2[p] += g.detach().pow(2)
mu  = {p: sum_g[p]  / N for p in sum_g}
var = {p: sum_g2[p] / N - mu[p].pow(2) for p in sum_g}
gsnr = {p: mu[p].pow(2) / (var[p] + eps) for p in mu}
gsnr_global = torch.cat([g.flatten() for g in gsnr.values()]).mean()
```

La versión vectorizada sustituye el bucle por `torch.func.vmap(grad(loss_fn))` aplicado al lote completo de $N$ ejemplos. La agregación final produce un escalar global (media sobre parámetros) y un escalar por capa.

**Consideraciones.** El paper liga GSNR alto a *gap* de generalización pequeño y, por tanto, a mejor *test loss*; este vínculo justifica su uso como predictor temprano. La métrica varía sensiblemente entre capas y entre pasos, por lo que conviene registrar ambas dimensiones. En CIFAR-10 con ResNet el coste de los gradientes per-example es relevante, lo que motiva submuestrear $N$ moderado ($N \in [256, 1024]$) sobre un subconjunto fijo a lo largo del entrenamiento para garantizar comparabilidad entre épocas.

**Logging.** Se registra el GSNR medio global y por capa en cada época, junto con la *train loss* y la *test loss* en el mismo paso, para alimentar el análisis de correlación con las métricas de eficiencia (epochs-to-threshold, AUC de *test loss*, mejor *test loss*). Opcionalmente se almacena un histograma por época del GSNR por parámetro para inspección cualitativa.

## Notes
- Métrica de **eje varianza**: directamente comparable con normalized gradient variance y con el estadístico de Faghri.
- A diferencia de Faghri (centrado en velocidad de convergencia), GSNR ata varianza ↔ generalización con marco teórico explícito.
- Encaje en TFG: candidata fuerte como métrica temprana, fundamenta teóricamente por qué baja varianza relativa predice mejor rendimiento final.

## Cited By
