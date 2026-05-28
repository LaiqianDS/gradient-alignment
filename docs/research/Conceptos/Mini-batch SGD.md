
Mini-batch SGD estima el gradiente sobre un subconjunto $S$ de $B$ ejemplos extraídos del dataset en cada paso, y actualiza los parámetros con $w_{t+1} = w_t - \eta_t \cdot \tfrac{1}{B}\sum_{i\in S_t}\nabla\ell_i(w_t)$. Es el punto medio entre el gradiente full-batch, que es caro pero exacto, y SGD puro sobre un solo ejemplo, que es barato pero extremadamente ruidoso. En la práctica todo el deep learning moderno corre alguna variante de mini-batch.

El tamaño de batch controla un trade-off muy concreto. La [[Varianza del gradiente]] del estimador escala $1/B$, así que duplicar $B$ reduce el ruido a la mitad y permite usar un learning rate proporcionalmente mayor, lo que en la práctica significa la misma trayectoria efectiva con la mitad de pasos. Pero esta linealidad solo se mantiene hasta el [[Batch size crítico]]. A partir de ahí, el gradiente medio ya domina al ruido, y añadir más ejemplos solo gasta cómputo sin acelerar la convergencia en pasos.

Un ejemplo numérico ayuda. Con MNIST y sus 60 000 ejemplos, un batch de 100 ve 1/600 del dataset por paso y necesitas 600 iteraciones por epoch. Si subes a batch 1000, son solo 60 iteraciones por epoch pero cada una procesa diez veces más datos. El cómputo total es el mismo. Lo que cambia es si tu GPU puede paralelizar esos diez ejemplos simultáneamente: si sí, ahorras tiempo real; si no (porque el batch ya satura la memoria), no ahorras nada. Por eso en práctica $B$ se elige por aprovechamiento del hardware, típicamente potencias de 2 que llenan la GPU sin desbordarla.

Hay un detalle teórico importante sobre por qué mini-batch funciona donde batch puro no escala. El gradiente full-batch sobre 60 000 ejemplos requiere mantener todas las activaciones en memoria, lo que es prohibitivo para redes grandes. SGD con un solo ejemplo es viable pero su varianza es tan alta que cada paso parece aleatorio. Mini-batch es el compromiso que existe precisamente porque la varianza decrece como $1/B$, así que un $B$ moderado (32, 64, 128, 256) ya da una estimación suficientemente fiable para que los pasos individuales tengan dirección útil.

## Enlaces

- Trade-off paralelismo-ruido y batch crítico: [[An Empirical Model of Large-Batch Training]]
- Descomposición de la varianza por batch en redes reales: [[A Study of Gradient Variance in Deep Learning]]
- Panorama de variantes (SGD, mini-batch, batch full): [[An overview of gradient descent optimization algorithms]]
