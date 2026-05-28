
El signal-to-noise ratio del gradiente es el cociente entre cuánta señal coherente hay en el gradiente y cuánto ruido inter-muestra lo rodea. La definición operacional cambia según el grano al que mires.

Por parámetro, lo que se llama GSNR (gradient signal-to-noise ratio), es $r(\theta_j) = \tilde g(\theta_j)^2 / \rho^2(\theta_j)$, con $\tilde g(\theta_j) = \mathbb{E}_x[\nabla_{\theta_j}\ell(x;\theta)]$ la media del gradiente sobre la distribución de datos y $\rho^2(\theta_j) = \mathrm{Var}_x[\nabla_{\theta_j}\ell(x;\theta)]$ su varianza. Mide, parámetro a parámetro, si la señal compartida entre ejemplos domina al ruido. Como vector global se escribe $\mathrm{SNR} = \|\mathbb{E}[g]\|^2 / \mathrm{tr}(\mathrm{Cov}(g))$, que es exactamente el recíproco de la varianza normalizada del gradiente y aparece como predictor del [[Batch size crítico]] y de la velocidad de convergencia.

Un ejemplo concreto de por qué la magnitud importa. Imagina dos escenarios al entrenar una red de clasificación. En el primero, todas las imágenes de la clase "gato" empujan el peso $\theta_j$ en sentido positivo (la derivada respecto a $\theta_j$ es positiva para todas) con valor medio $0.01$ y desviación entre ejemplos $0.003$. El GSNR es aproximadamente $0.01^2 / 0.003^2 \approx 11$, así que el gradiente promedio es 3 veces más grande que su error típico, y dos veces "señal" por cada "ruido" al cuadrado. En el segundo escenario, la mitad de las imágenes empujan positivo y la otra mitad negativo con la misma magnitud. La media es prácticamente cero, la varianza está intacta, y el GSNR colapsa a algo cercano a cero. Visto por parámetro, este segundo régimen es el sello de la memorización: cada ejemplo "tira" por separado de los pesos sin que la red consolide un patrón común.

El interés práctico es que un GSNR alto durante el entrenamiento se asocia empíricamente con un [[Gap de generalización]] pequeño. La intuición es que un GSNR alto significa que la red está aprendiendo cosas que muchos ejemplos comparten, no idiosincrasias individuales, así que esas direcciones de aprendizaje son estables y por tanto generalizables. Por eso GSNR es un proxy train-time de generalización. Adam usa una versión por parámetro como mecanismo interno: el ratio $\hat m / \sqrt{\hat v}$ es un SNR efectivo que atenúa el paso cuando crece la incertidumbre direccional, lo que actúa como un annealing automático sin necesidad de schedule.

## Enlaces

- GSNR por parámetro y conexión con generalización: [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters]]
- SNR global y su relación con batch size: [[An Empirical Model of Large-Batch Training]]
- Varianza del gradiente como denominador del SNR: [[A Study of Gradient Variance in Deep Learning]]
- $\hat m/\sqrt{\hat v}$ como SNR por parámetro en Adam: [[Adam - A Method for Stochastic Optimization]]
