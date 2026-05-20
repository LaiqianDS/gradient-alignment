---
authors:
  - Sam McCandlish
  - Jared Kaplan
  - Dario Amodei
year: 2018
status: read
relevance: medium
last_review: 2026-05-07
url: https://arxiv.org/pdf/1812.06162
---

# An Empirical Model of Large-Batch Training

## Summary

### Contextualización

El paper aborda un problema central en deep learning a escala: el trade-off entre el tamaño de batch utilizado durante el entrenamiento, el grado de paralelismo de datos alcanzable y la eficiencia computacional total. La motivación parte de una observación empírica acumulada: el límite a partir del cual ampliar el batch deja de aportar speed-up casi lineal varía radicalmente según el dominio. En clasificación sobre ImageNet se han reportado batch sizes efectivos del orden de decenas de miles (8k, 16k, 32k e incluso 64k); en modelos de lenguaje el orden es de miles; mientras que el agente Dota de OpenAI utilizó batches de millones de timesteps. Antes de este trabajo, no existía un marco conceptual unificado que explicase de forma cuantitativa por qué estos límites difieren entre tareas ni cómo anticipar el batch size adecuado para un problema nuevo. El paralelismo de datos requiere comunicación rápida entre dispositivos, pero su utilidad real depende de que el batch grande siga siendo algorítmicamente efectivo: si el ruido del gradiente ya es bajo, duplicar el batch apenas mejora el paso de optimización y desperdicia cómputo.

### Aportación principal

La contribución central es introducir el gradient noise scale como predictor empírico, simple de medir, del batch size crítico (B_crit) por encima del cual la eficiencia de paralelización cae significativamente. Los autores derivan, partiendo de hipótesis básicas sobre la función de pérdida y la covarianza del gradiente por ejemplo, una relación que permite predecir tanto la forma exacta de la curva de Pareto entre tiempo de entrenamiento (S, número de pasos) y cómputo total (E, ejemplos procesados) como el punto de inflexión donde el paralelismo deja de ser productivo. El gradient noise scale es esencialmente una medida de la relación señal/ruido del gradiente entre ejemplos del dataset, y su valor coincide aproximadamente con el batch crítico para una amplia variedad de tareas, optimizadores y arquitecturas.

### Metodología

La derivación parte de considerar el gradiente estimado $G_{\text{est}}$ sobre un batch de tamaño $B$ como variable aleatoria con esperanza igual al gradiente verdadero $G$ y covarianza $\Sigma / B$, donde $\Sigma$ es la covarianza por ejemplo. Una expansión de Taylor de segundo orden de la pérdida bajo un paso de tamaño $\epsilon$ en la dirección de $G_{\text{est}}$ da la mejora esperada de la pérdida. Optimizando respecto a $\epsilon$, se obtiene que el step size óptimo decae como

$$\epsilon_{\text{opt}}(B) = \frac{\epsilon_{\max}}{1 + \mathcal{B}_{\text{noise}} / B},$$

y la mejora óptima en pérdida tiene la misma forma funcional. De aquí emerge la definición exacta del noise scale ponderado por la Hessiana:

$$\mathcal{B}_{\text{noise}} = \frac{\mathrm{tr}(H \Sigma)}{G^\top H G}.$$

Bajo el supuesto (idealizado) de que la Hessiana es proporcional a la identidad, la ecuación se reduce a la forma simplificada que se usa en la mayoría de experimentos:

$$\mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|G\|^2}.$$

Esto es: la suma de las varianzas de las componentes del gradiente dividida por la norma global al cuadrado del gradiente. Es una métrica de cuán grande es el gradiente respecto a su variabilidad entre ejemplos. Los autores muestran que en distancia $L^2$ normalizada, $\mathbb{E}[\|G_{\text{est}} - G\|^2] / \|G\|^2 = \mathcal{B}_{\text{simple}} / B$, lo que conecta directamente la calidad del estimado de batch con la métrica.

A partir de promediar la mejora local sobre toda la trayectoria de entrenamiento, se obtiene la ecuación clave del trade-off (forma hiperbólica):

$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1,$$

donde $S_{\min}$ es el número mínimo de pasos posible (límite de batch infinito) y $E_{\min}$ el mínimo de ejemplos procesados (límite de batch unitario). Definen el batch size crítico empírico como $\mathcal{B}_{\text{crit}} = E_{\min} / S_{\min}$, que el modelo predice aproximadamente igual al noise scale promediado adecuadamente.

Los autores distinguen dos versiones de la métrica: el noise scale verdadero ($\mathcal{B}_{\text{noise}}$, que requiere productos Hessiano-vector) y el simple noise scale ($\mathcal{B}_{\text{simple}}$), computacionalmente mucho más barato. Demuestran que ambos suelen diferir sólo por un factor multiplicativo pequeño, especialmente cuando se usan esquemas que mejoran el condicionamiento. Además proporcionan en el apéndice A.1 un método sin overhead para estimar $\mathcal{B}_{\text{simple}}$ en entornos de entrenamiento data-paralelo, comparando la norma del gradiente local (antes de promediar entre dispositivos) con la norma del gradiente global (después de promediar) y combinándolas con medias móviles exponenciales.

### Datasets y modelos

La verificación empírica cubre ocho tareas distribuidas en tres paradigmas de aprendizaje:

- **Image classification**: MNIST (CNN simple con SGD), SVHN (CNN simple con SGD y Adam, también con autoencoder y VAE para generative modeling), CIFAR-10 (ResNet-32 con momentum), ImageNet (ResNet-50 con momentum y schedule de learning rate decay).
- **Reinforcement learning**: siete juegos de Atari (Alien, Beam Rider, Breakout, Pong, Qbert, Seaquest, Space Invaders) entrenados con A2C y RMSProp; agentes de Dota 1v1 y 5v5 entrenados con PPO asíncrono por el equipo OpenAI Dota.
- **Language modeling**: LSTM autoregresivo de tamaño 2048 (también 1024 y 512 para estudio de dependencia con el tamaño del modelo) sobre el Billion Word Benchmark con codificación BPE de vocabulario 40k, optimizado con Adam.
- **Generative modeling**: Variational Autoencoder con arquitectura InfoGAN sobre SVHN y un autoencoder simple de la misma arquitectura.

### Métricas

Las métricas centrales son: training steps ($S$), ejemplos procesados ($E$) hasta alcanzar un nivel objetivo de rendimiento (error de clasificación, perplejidad, recompensa de episodio o TrueSkill en Dota), y la curva de Pareto entre ambos. Para cada tarea se hace una grid search sobre batch size y learning rate, se ajustan los frentes de Pareto a la ecuación hiperbólica, y se mide $\mathcal{B}_{\text{crit}}$ como $E_{\min}/S_{\min}$ para varios niveles objetivo. Paralelamente se mide $\mathcal{B}_{\text{simple}}$ durante el entrenamiento. El criterio operativo de batch crítico se define como el batch al que la eficiencia de cómputo cae al 50% del óptimo, punto en el cual el training speed también es típicamente el 50% del máximo.

### Resultados principales

El simple noise scale predice $\mathcal{B}_{\text{crit}}$ al nivel de orden de magnitud en todos los dominios estudiados, con valores que van de unos 20 (autoencoder SVHN) hasta más de 10 millones (Dota 5v5). La concordancia es notable considerando que la fórmula se deriva localmente y se aplica a trayectorias completas. Los frentes de Pareto se ajustan muy bien a la ecuación $(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$. El learning rate óptimo sigue la forma funcional $\epsilon_{\text{opt}}(B) = \epsilon_{\max} / (1 + \mathcal{B}_*/B)$ para SGD/momentum ($\alpha = 1$), mientras que para Adam y RMSProp se observa una power law con $\alpha \in [0.5, 1.0]$.

Otras observaciones empíricas relevantes:

- El noise scale crece a lo largo del entrenamiento, típicamente en uno o más órdenes de magnitud, porque $\|G\|$ disminuye conforme se aproxima un mínimo mientras $\mathrm{tr}(\Sigma)$ permanece relativamente constante. Esto implica que el batch crítico también depende del nivel de rendimiento al que se entrena.
- A pérdida fija, el noise scale es aproximadamente independiente del tamaño del modelo (verificado con LSTMs de 512, 1024 y 2048). Modelos más grandes alcanzan mayor noise scale solo porque alcanzan menor pérdida.
- Tareas más complejas presentan mayor noise scale: Dota 5v5 supera a Dota 1v1, RL supera a clasificación, y dentro de Atari, Pong (más simple) tiene noise scale menor que Beam Rider o Space Invaders.
- Los modelos generativos sobre SVHN tienen $\mathcal{B}_{\text{simple}}$ significativamente menor que los clasificadores sobre el mismo dataset, sugiriendo que extraen más información por ejemplo.
- El noise scale depende del learning rate vía una "temperatura" de entrenamiento $T = \epsilon / \epsilon_{\max}(B)$, con $\mathcal{B}_{\text{noise}}$ proporcional a $1/T$ en equilibrio. Esto explica por qué runs bien tuneadas con diferentes batches presentan el mismo noise scale al mismo nivel de pérdida.
- Variar dinámicamente el batch size proporcionalmente a $\sqrt{\mathcal{B}_{\text{simple}}}$ durante el entrenamiento produce mejoras modestas en eficiencia (estimadas en el orden del 4% para SVHN), aunque experimentos preliminares sugieren beneficios potencialmente mayores.

### Conclusiones

El gradient noise scale, en su forma simplificada $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma) / \|G\|^2$, es un predictor empíricamente sólido (al nivel de orden de magnitud) del batch size crítico que separa el régimen de paralelización lineal del régimen de retornos decrecientes. Esto permite estimar el batch size útil para una tarea nueva a partir de un único entrenamiento (incluso parcial), reduciendo drásticamente el coste de las búsquedas por ensayo y error. La curva de Pareto entre pasos y ejemplos procesados sigue una forma hiperbólica universal, y el batch crítico se interpreta como el punto en que se duplica simultáneamente el cómputo mínimo y el tiempo mínimo: un compromiso natural entre eficiencia temporal y computacional. La complejidad de la tarea se refleja en el noise scale, lo que sugiere que problemas más ricos (RL complejo, lenguaje) son más paralelizables. El trabajo proporciona, además, una explicación parcial del rol de la temperatura de entrenamiento y motivación teórica para schedules adaptativos de batch size, aunque su tratamiento se centra deliberadamente en B asumiendo que el learning rate se tunea posteriormente.

## Medición y pipeline

**Métrica concreta.** Se adopta el simple gradient noise scale de McCandlish et al. en su forma idealizada (Hessiana proporcional a la identidad):

$$\mathcal{B}_{\text{simple}} = \frac{\mathrm{tr}(\Sigma)}{\|G\|^2},$$

donde $\Sigma$ es la covarianza por ejemplo del gradiente estocástico y $G$ el gradiente verdadero (esperanza sobre el dataset). En la práctica, ambos términos se estiman sin necesidad de productos Hessiano-vector mediante el estimador de dos batch sizes $B_{\text{small}} < B_{\text{big}}$ (apéndice A.1 del paper). Sea $G_{\text{small}}$ y $G_{\text{big}}$ los gradientes promedio sobre lotes disjuntos de tamaño $B_{\text{small}}$ y $B_{\text{big}}$ respectivamente. Los estimadores insesgados son:

$$\|G\|^2 \approx \mathcal{G}_{\text{est}} = \frac{B_{\text{big}} \cdot \|G_{\text{big}}\|^2 - B_{\text{small}} \cdot \|G_{\text{small}}\|^2}{B_{\text{big}} - B_{\text{small}}},$$

$$\mathrm{tr}(\Sigma) \approx \mathcal{S}_{\text{est}} = \frac{\|G_{\text{small}}\|^2 - \|G_{\text{big}}\|^2}{1/B_{\text{small}} - 1/B_{\text{big}}},$$

y $\mathcal{B}_{\text{simple}} \approx \mathcal{S}_{\text{est}} / \mathcal{G}_{\text{est}}$. Para reducir varianza se aplican medias móviles exponenciales sobre numerador y denominador antes de dividir.

**Entradas.** Dos estimaciones del gradiente en el mismo $w$ usando lotes disjuntos de tamaños $B_{\text{small}}$ y $B_{\text{big}}$ (típicamente $B_{\text{big}} = k \cdot B_{\text{small}}$ con $k \in [4, 8]$). Alternativa más precisa pero costosa: per-example gradients vía `vmap`, calculando $\mathrm{tr}(\Sigma)$ directamente como varianza muestral entre ejemplos.

**Cuándo computar.** Una medición por época, o cada $K$ pasos ($K \in [100, 500]$) si las épocas son largas. La frecuencia debe ser suficiente para capturar el crecimiento monótono de $\mathcal{B}_{\text{simple}}$ observado en el paper.

**Coste.** Dos backward passes adicionales (uno con $B_{\text{small}}$, uno con $B_{\text{big}}$) más el cálculo de sus normas $L^2$; despreciable frente al coste del paso de entrenamiento. La variante per-sample con `vmap` multiplica el coste del backward por el tamaño del lote y se reserva como validación puntual.

**Integración en el pipeline.**

```python
def gns_simple(model, loss_fn, batch_small, batch_big):
    g_small = flat_grad(model, loss_fn, batch_small)
    g_big   = flat_grad(model, loss_fn, batch_big)
    n_s, n_b = g_small.pow(2).sum(), g_big.pow(2).sum()
    B_s, B_b = len(batch_small), len(batch_big)
    G_est = (B_b * n_b - B_s * n_s) / (B_b - B_s)
    S_est = (n_s - n_b) / (1.0 / B_s - 1.0 / B_b)
    return (S_est / G_est).item()  # log como escalar por época
```

**Consideraciones.** $\mathcal{B}_{\text{simple}}$ predice el batch size crítico y, por tanto, la paralelizabilidad esperada del entrenamiento; resulta candidato natural como variable explicativa de la eficiencia de cómputo total. El paper observa que crece uno o más órdenes de magnitud durante el entrenamiento ($\|G\|$ disminuye, $\mathrm{tr}(\Sigma)$ permanece estable), por lo que conviene registrar su trayectoria, no sólo un valor puntual. El cómputo per-layer es informativo —capas con GNS distinto sugieren heterogeneidad en la dinámica— y guarda consistencia conceptual con la varianza normalizada del gradiente (NGV) de Faghri et al., que mide ruido relativo por parámetro.

**Logging.** $\mathcal{B}_{\text{simple}}$ como escalar por época (clave `gns/simple`); opcionalmente $\mathcal{B}_{\text{simple}}$ per layer como diccionario por módulo (clave `gns/per_layer/<name>`). Guardar también las normas intermedias $\|G_{\text{small}}\|^2$ y $\|G_{\text{big}}\|^2$ facilita auditoría posterior y permite recalcular el estimador con diferentes ventanas de EMA.

## Notes
Es un ejemplo de cómo una métrica relacionada con el ruido del gradiente permite seleccionar un valor para un hiperparámetro, en este caso batch size, permitiendo aumentar paralelismo y por ende reducir tiempos, mientras se conserva el rendimiento.

Parten de que se puede utilizar cierto batch size grande sin perjudicar al rendimiento, pero entre problemas, este batch size no es el mismo.
