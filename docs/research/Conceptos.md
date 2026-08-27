# Conceptos

Glosario del TFG: una sección por concepto, agrupadas por tema. Los enlaces internos `[[#Concepto]]` apuntan a otra entrada de este mismo documento; la línea **Papers** de cada entrada apunta a las notas de `Papers/` que lo fundamentan.

**Alineación direccional de gradientes:** [[#Gradientes per-sample]] · [[#Similitud coseno entre gradientes]] · [[#Coherencia de gradientes]] · [[#Gradient confusion]] · [[#Stiffness]] **Varianza estocástica y batch:** [[#Varianza del gradiente]] · [[#Mini-batch SGD]] · [[#Estimador insesgado del gradiente]] · [[#SNR del gradiente]] · [[#Batch size crítico]] **Optimización (paso, momento, adaptatividad):** [[#Momentum]] · [[#Primer y segundo momento del gradiente]] · [[#Tasa de aprendizaje adaptativa]] · [[#LR decay]] **Generalización y features:** [[#Sobreparametrización]] · [[#Feature learning]] · [[#Memorización vs generalización]] · [[#Gap de generalización]] · [[#Early stopping]] · [[#Proxy de generalización train-time]]

---

## Alineación direccional de gradientes

### Gradientes per-sample

El gradiente per-sample $g_i = \nabla_W \ell(x_i, y_i; W)$ es el gradiente de la pérdida para un único ejemplo, antes de promediar sobre el minibatch. Sobre él se construyen casi todas las métricas de alineación, coherencia y varianza, porque toda la información estadística se pierde al reducir los $g_i$ a su media $\hat g = \tfrac{1}{B}\sum_i g_i$. Su distribución (covarianza, asimetría, curtosis) describe cuánto ruido hay; sus productos internos por pares describen si los ejemplos cooperan o se cancelan ([[#Stiffness]], m-coherence, [[#Gradient confusion]]); y su similitud con cantidades de referencia da pie a proxies de generalización como GWA (gradiente per-sample contra los pesos del clasificador).

Un ejemplo del cambio de granularidad. En un batch de 128 imágenes de CIFAR-10 el gradiente medio es un único vector en $\mathbb{R}^p$ con $p$ del orden de millones; los gradientes per-sample son una matriz $128 \times p$. Con la matriz de Gram $G_{ij} = \langle g_i, g_j\rangle$, de tamaño $128 \times 128$, se lee directamente qué ejemplos están alineados (entrada grande positiva), cuáles son ortogonales (entrada cercana a cero) y cuáles entran en conflicto (entrada negativa). En PyTorch esos gradientes se vectorizan con `torch.func.vmap(grad(loss_fn))`; sin esa primitiva haría falta un forward-backward por ejemplo, 128 veces más caro, y por eso medir alineación durante el entrenamiento no fue viable hasta hace poco.

**Papers:** [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]], [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Chatterjee 2020]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]], [[Stiffness - A New Perspective on Generalization in Neural Networks|Fort et al. 2019]]

### Similitud coseno entre gradientes

El coseno entre dos gradientes $\cos(g_1, g_2) = \langle g_1, g_2\rangle / (\|g_1\|\,\|g_2\|)$ es la medida elemental de alineación direccional. Vale $1$ si los dos vectores apuntan en la misma dirección, $0$ si son ortogonales y $-1$ si son opuestos. Se usa en lugar del producto interno crudo porque normaliza por magnitud. Con $g_1 = (10, 0)$ y $g_2 = (1, 1)$ el producto interno vale $10$, cifra que viene casi entera de que $g_1$ es muy grande y no de que los dos vectores compartan dirección; el coseno da $10/(10\sqrt{2}) \approx 0.707$, que captura honestamente el ángulo de 45 grados. Para coherencia y conflicto lo que importa es el ángulo, no la escala.

La misma operación aparece con agregadores distintos sobre los [[#Gradientes per-sample]]: promedios por pares intra-clase e inter-clase en [[#Stiffness]], el coseno más negativo de cualquier par en [[#Gradient confusion]], el coseno contra los pesos del clasificador final en GWA, y una suma de cosenos por pares implícita en m-coherence al agregar $\|\sum g_i\|^2 / \sum \|g_i\|^2$.

Una sutileza para el uso práctico. En alta dimensión el coseno entre vectores aleatorios se concentra cerca de cero. Si los $g_i$ tienen $p$ componentes y son ruido isotrópico, el coseno típico entre dos cualesquiera es $O(1/\sqrt{p})$. Cualquier desviación positiva sistemática del coseno medio es por tanto señal real, no fluctuación. En una red con $p \sim 10^7$, un coseno promedio de $0.05$ ya es enorme, equivalente a varios cientos de desviaciones por encima de lo esperable por azar.

**Papers:** [[Stiffness - A New Perspective on Generalization in Neural Networks|Fort et al. 2019]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]], [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent|Sankararaman et al. 2020]]

### Coherencia de gradientes

La hipótesis de gradientes coherentes (CGH) postula que cuando ejemplos similares tienen gradientes que apuntan en direcciones similares, el gradiente del minibatch se refuerza en esas direcciones compartidas y se cancela en las idiosincráticas. Explica por qué SGD generaliza sin overfitear catastróficamente. Las direcciones reforzadas por muchos ejemplos son estables ante la remoción de uno, y por estabilidad algorítmica esa estabilidad implica buena generalización.

Formalmente, la norma del gradiente del minibatch se descompone como $\|g\|^2 = \sum_e \|g^e\|^2 + \sum_{e\neq e'}\langle g^e, g^{e'}\rangle$. El primer término es la "energía individual" de cada ejemplo, y el segundo es el término cruzado donde vive la coherencia. Si los productos internos son típicamente positivos, el gradiente agregado crece superlinealmente con el número de ejemplos y SGD se mueve en direcciones que sirven a muchos a la vez. Si son cercanos a cero, los ejemplos son independientes y el gradiente medio crece como $\sqrt{m}$ (ruido aditivo). Si son típicamente negativos, lo que se llama [[#Gradient confusion]], el agregado puede ser más pequeño que cualquiera de las contribuciones individuales.

La métrica operacional más limpia es m-coherence: $\alpha_m = \|\sum_i g_i\|^2 / \sum_i \|g_i\|^2 \in [0, m]$. Vale $1$ en el límite ortogonal (gradientes perpendiculares de misma norma), $m$ si son perfectamente paralelos, y por debajo de $1$ si están anticorrelados. Se lee como el número medio de ejemplos que se benefician de un paso de descenso a lo largo del gradiente de un ejemplo elegido al azar. En un minibatch de 128 imágenes de CIFAR-10 con etiquetas reales, $\alpha_m$ típicamente vale entre 5 y 20; con etiquetas aleatorias baja a $\alpha_m \approx 1.1$, es decir, cada paso ayuda solo al ejemplo de cuyo gradiente vino. Las dos trayectorias de pérdida acaban pareciéndose, pero la coherencia las separa desde el principio, y eso es barato de medir durante el entrenamiento (ver también [[#Memorización vs generalización]]).

**Papers:** [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Chatterjee 2020]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]], [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]]

### Gradient confusion

Gradient confusion es una cota inferior $\eta \geq 0$ tal que $\langle \nabla f_i(w), \nabla f_j(w)\rangle \geq -\eta$ para todo par de [[#Gradientes per-sample]]. Mide, en el peor caso, cuán antagónicos llegan a ser los gradientes individuales: si $\eta$ es pequeña, los ejemplos cooperan; si es grande, hay pares que empujan con fuerza en direcciones opuestas y SGD avanza muy poco en la dirección agregada. Es el extremo negativo de la [[#Coherencia de gradientes]].

Conecta directamente con la teoría de convergencia clásica. Bajo suavidad de Lipschitz (gradiente $L$-suave) y la condición Polyak-Lojasiewicz que generaliza la convexidad fuerte, SGD con paso constante converge linealmente a un entorno del óptimo cuyo radio escala con $\eta$. La fórmula típica es $\mathbb{E}\|\nabla F(w_T)\|^2 \leq \rho(F(w_1)-F^*)/T + \rho \eta$, así que cuando $T$ es grande lo que queda es un "suelo" de pérdida proporcional a la confusion. Reducir $\eta$ es lo que permite a SGD acercarse más al óptimo sin decaer el learning rate.

Empíricamente crece con la profundidad, porque dos imágenes parecidas al entrar producen activaciones cada vez más distintas al atravesar más capas no lineales. En cambio crece poco o incluso decrece con la anchura, que es una de las explicaciones formales de por qué la [[#Sobreparametrización]] ayuda a SGD. En redes muy anchas el espacio de gradientes es tan grande que los ejemplos casi nunca compiten. Batch normalization y skip connections la reducen mucho, y una ResNet la mantiene moderada a profundidades donde una red plana equivalente entraría en confusion catastrófica antes de poder entrenarse.

**Papers:** [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent|Sankararaman et al. 2020]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]]

### Stiffness

Stiffness es el coseno (o, en su variante más cruda, la coincidencia de signo) entre los [[#Gradientes per-sample]] de dos ejemplos: $S(i,j) = \cos(g_i, g_j)$. Si es positiva, un paso de gradiente diseñado para reducir la pérdida sobre $i$ también la reduce sobre $j$, lo que significa que la red ha aprendido un feature compartido que sirve a ambos. Si es negativa, mejorar uno empeora el otro y la red oscila entre dos objetivos incompatibles.

Lo más informativo no es la stiffness global sino su desagregación. La matriz de class stiffness $C(c_a, c_b) = \mathbb{E}_{x_1\in c_a, x_2\in c_b}[S(x_1, x_2)]$ promedia sobre pares de ejemplos de las clases $c_a, c_b$. La diagonal mide generalización intra-clase y las entradas extra-diagonales miden transferencia entre clases, revelando jerarquías semánticas que la red ha descubierto sola. En CIFAR-10, $C(\text{gato}, \text{perro})$ es claramente mayor que $C(\text{gato}, \text{camión})$, porque la red ha aprendido features que sirven para animales en general, así que mejorar en gatos mejora un poco en perros y solo accidentalmente en camiones.

También se mide frente a la distancia entre los inputs $x_i, x_j$ en el espacio original. La dynamic critical length $\xi$ es la distancia a la que la stiffness intra-clase cruza cero según un ajuste lineal a la curva stiffness-distancia, y cuantifica el tamaño típico del "parche" del espacio de inputs que se mueve junto bajo un paso de gradiente. Empieza pequeña, porque la red trata cada ejemplo como aislado, y crece a medida que aprende features globales, lo que es una firma directa de [[#Feature learning]]. En MNIST $\xi$ pasa de prácticamente cero al inicio a varios pixels de radio efectivo al final. Stiffness, m-coherence y GWA son el mismo coseno con agregadores distintos, correlacionan fuerte entre sí, y la elección entre ellas suele ser práctica más que teórica.

**Papers:** [[Stiffness - A New Perspective on Generalization in Neural Networks|Fort et al. 2019]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]]

---

## Varianza estocástica y batch

### Varianza del gradiente

En SGD, el gradiente de cada paso es un estimador $\hat g$ del gradiente real $\nabla f$ obtenido sobre un minibatch. Su varianza no es residual sino estructural, y persiste incluso en el óptimo, donde $\nabla f = 0$ pero $\mathrm{Var}[\hat g] > 0$, porque cada minibatch apunta a un objetivo ligeramente distinto y las contribuciones individuales no se cancelan al promediarlas. Es lo que obliga a decaer el learning rate ([[#LR decay]]), lo que define el [[#Batch size crítico]] y lo que está acoplado inversamente con la [[#Coherencia de gradientes]]. Si los ejemplos están bien alineados, la varianza del promedio es baja y la señal domina.

Para un minibatch de tamaño $B$ con ejemplos i.i.d., la varianza del estimador es $\mathrm{Var}[\hat g] = \tfrac{1}{B}\,\mathrm{tr}(\Sigma)$ con $\Sigma$ la covarianza per-sample. Escala $1/B$, así que reduce el ruido pero no lo elimina. Entrenando en CIFAR-10 con batch de 128, cada paso ve menos del 0.3 por ciento del dataset y la nube de ruido alrededor del gradiente medio tiene desviación típica del mismo orden que la señal cerca del óptimo; subir a batch 512 solo divide esa desviación por 2. El suelo de ruido viene de que los ejemplos no son redundantes.

**Papers:** [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]], [[An Empirical Model of Large-Batch Training|McCandlish et al. 2018]], [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction|Johnson & Zhang 2013]], [[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning|Defazio & Bottou 2019]]

### Mini-batch SGD

Mini-batch SGD estima el gradiente sobre un subconjunto $S$ de $B$ ejemplos y actualiza los parámetros con $w_{t+1} = w_t - \eta_t \cdot \tfrac{1}{B}\sum_{i\in S_t}\nabla\ell_i(w_t)$. Es el punto medio entre el gradiente full-batch, exacto pero que exige mantener en memoria las activaciones de todo el dataset, y SGD sobre un solo ejemplo, barato pero tan ruidoso que cada paso parece aleatorio. Funciona porque la [[#Varianza del gradiente]] decrece como $1/B$, y un $B$ moderado (32, 64, 128, 256) ya da pasos con dirección útil.

El tamaño de batch controla un trade-off concreto. Duplicar $B$ reduce el ruido a la mitad y permite un learning rate proporcionalmente mayor, es decir, la misma trayectoria efectiva con la mitad de pasos. Esa linealidad solo se mantiene hasta el [[#Batch size crítico]]; por encima, el gradiente medio ya domina al ruido y añadir ejemplos solo gasta cómputo.

Con MNIST y sus 60 000 ejemplos, un batch de 100 necesita 600 iteraciones por epoch y uno de 1000 solo 60, pero cada una procesa diez veces más datos, así que el cómputo total es el mismo. Lo que cambia es si la GPU puede paralelizar esos diez ejemplos a la vez; si el batch ya satura la memoria, no se ahorra nada. Por eso en la práctica $B$ se elige por aprovechamiento del hardware, típicamente potencias de 2 que llenan la GPU sin desbordarla.

**Papers:** [[An Empirical Model of Large-Batch Training|McCandlish et al. 2018]], [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]], [[An overview of gradient descent optimization algorithms|Ruder 2017]]

### Estimador insesgado del gradiente

Un estimador estocástico del gradiente es insesgado si $\mathbb{E}[\hat g] = \nabla f$, es decir, si su esperanza coincide con el gradiente que se querría calcular. Es la propiedad mínima que garantiza que SGD optimiza la pérdida correcta y no una distorsión sistemática de ella. [[#Mini-batch SGD]] la cumple cuando los ejemplos se muestrean i.i.d. de la distribución empírica. Si $f(w) = \tfrac{1}{n}\sum_{i=1}^n \ell_i(w)$, el estimador $\hat g = \tfrac{1}{B}\sum_{i\in S}\nabla \ell_i(w)$ con $S$ uniforme tiene por linealidad la esperanza del gradiente full-batch. De ahí salen las garantías clásicas de convergencia, incluso con paso constante a un entorno del óptimo.

Variantes como SVRG mantienen el insesgamiento con un truco de control variate: $\hat g_{\text{SVRG}} = \nabla\ell_i(w) - \nabla\ell_i(\tilde w) + \tilde\mu$, donde $\tilde w$ es un snapshot congelado del modelo y $\tilde\mu$ su gradiente full-batch precalculado. La esperanza del término que se resta es exactamente $\tilde\mu$, así que la corrección no introduce sesgo, pero reduce drásticamente la varianza cuando el iterado está cerca del snapshot.

El insesgamiento se rompe cuando se pierde la estructura finite-sum. Con batch normalization, las estadísticas de normalización dependen del minibatch que se está procesando, así que $\nabla\ell_i$ no es la misma función para distintos batches; lo mismo pasa con data augmentation aleatoria y con dropout. En esos casos $\nabla\ell_i$ deja de ser una función pura del peso, lo que invalida los supuestos de los métodos de reducción de varianza y explica por qué SVRG fracasa en redes profundas modernas a pesar de su superioridad teórica en el caso convexo.

**Papers:** [[Accelerating Stochastic Gradient Descent using Predictive Variance Reduction|Johnson & Zhang 2013]], [[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning|Defazio & Bottou 2019]], [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]]

### SNR del gradiente

El signal-to-noise ratio del gradiente es el cociente entre la señal coherente del gradiente y el ruido inter-muestra que lo rodea, y su definición operacional cambia según el grano. Por parámetro, el GSNR (gradient signal-to-noise ratio) es $r(\theta_j) = \tilde g(\theta_j)^2 / \rho^2(\theta_j)$, con $\tilde g(\theta_j) = \mathbb{E}_x[\nabla_{\theta_j}\ell(x;\theta)]$ la media del gradiente sobre la distribución de datos y $\rho^2(\theta_j) = \mathrm{Var}_x[\nabla_{\theta_j}\ell(x;\theta)]$ su varianza. Como vector global se escribe $\mathrm{SNR} = \|\mathbb{E}[g]\|^2 / \mathrm{tr}(\mathrm{Cov}(g))$, que es el recíproco de la varianza normalizada del gradiente y aparece como predictor del [[#Batch size crítico]] y de la velocidad de convergencia.

Un ejemplo por parámetro. Si todas las imágenes de la clase "gato" empujan el peso $\theta_j$ en sentido positivo con valor medio $0.01$ y desviación entre ejemplos $0.003$, el GSNR vale $0.01^2 / 0.003^2 \approx 11$ y el gradiente promedio es 3 veces mayor que su error típico. Si en cambio la mitad de las imágenes empuja positivo y la otra mitad negativo con la misma magnitud, la media es prácticamente cero, la varianza sigue intacta y el GSNR colapsa. Ese segundo régimen es el sello de la memorización. Cada ejemplo empuja los pesos por separado sin que la red consolide un patrón común.

El interés práctico es que un GSNR alto durante el entrenamiento se asocia empíricamente con un [[#Gap de generalización]] pequeño. La red está aprendiendo cosas que muchos ejemplos comparten, y esas direcciones son estables y por tanto generalizables. Por eso el GSNR es un proxy train-time de generalización. Adam usa una versión por parámetro como mecanismo interno (ver [[#Tasa de aprendizaje adaptativa]]).

**Papers:** [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]], [[An Empirical Model of Large-Batch Training|McCandlish et al. 2018]], [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]], [[Adam - A Method for Stochastic Optimization|Kingma & Ba 2015]]

### Batch size crítico

El batch size crítico $\mathcal{B}_{\text{crit}}$ es el umbral a partir del cual aumentar el tamaño de batch deja de reducir el número de pasos necesarios para alcanzar un objetivo de pérdida. Por debajo, duplicar $B$ permite duplicar el learning rate (la regla de "scaling lineal") y el progreso por paso escala casi lineal con el cómputo. Por encima, la [[#Varianza del gradiente]] ya es lo bastante baja, el gradiente medio domina al ruido, y añadir ejemplos solo gasta cómputo.

Empíricamente, $\mathcal{B}_{\text{crit}}$ coincide con el gradient noise scale, en su versión simplificada $\mathcal{B}_{\text{simple}} = \mathrm{tr}(\Sigma) / \|G\|^2$, donde $\Sigma$ es la covarianza per-sample y $G$ el gradiente medio. Es exactamente el recíproco del [[#SNR del gradiente]] global. Si el SNR es alto, el batch crítico es bajo. La fórmula exacta pondera por curvatura usando la Hessiana $H$, $\mathcal{B}_{\text{noise}} = \mathrm{tr}(H\Sigma) / G^\top H G$, pero en la práctica se usa la simplificada porque calcular $H$ es prohibitivo en redes reales.

La relación funcional óptima entre el learning rate $\epsilon$ y el batch size $B$ satura según $\epsilon_{\text{opt}}(B) = \epsilon_{\max} / (1 + \mathcal{B}_{\text{noise}}/B)$. Para $B \ll \mathcal{B}_{\text{noise}}$ el denominador está dominado por $\mathcal{B}_{\text{noise}}/B$ y $\epsilon_{\text{opt}}$ escala lineal con $B$; para $B \gg \mathcal{B}_{\text{noise}}$ se satura en $\epsilon_{\max}$ y la regla lineal deja de aplicarse. Entrenando un modelo de lenguaje grande, $\mathcal{B}_{\text{simple}}$ vale del orden de $10^4$ a $10^6$ tokens, así que pasar de batch $10^6$ a batch $10^7$ solo ahorra un factor 2 en pasos, no un factor 10. La receta operacional es mirar si el $\epsilon$ óptimo sigue escalando linealmente al subir $B$. Si ya no lo hace, el batch pasó del crítico y el cómputo extra se aprovecha mejor en otra cosa.

**Papers:** [[An Empirical Model of Large-Batch Training|McCandlish et al. 2018]], [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]], [[On the Ineffectiveness of Variance Reduced Optimization for Deep Learning|Defazio & Bottou 2019]]

---

## Optimización (paso, momento, adaptatividad)

### Momentum

Momentum acumula una fracción del vector de actualización previo en la dirección de descenso: $v_t = \gamma v_{t-1} + \eta \nabla f(\theta_{t-1})$, $\theta_t = \theta_{t-1} - v_t$, típicamente con $\gamma = 0.9$. La intuición física es directa. Una bola que rueda colina abajo acumula velocidad donde la pendiente es consistente y se amortigua donde el terreno oscila. Eso resuelve dos problemas del descenso vanilla. El geométrico: en barrancos largos y estrechos, típicos cerca del óptimo, el gradiente apunta sobre todo a las paredes y SGD rebota de lado; momentum cancela el rebote, porque las contribuciones laterales de pasos sucesivos se promedian a cero, y acumula la componente longitudinal, que es constante en signo, lo que en casos cuadráticos mejora la tasa de $\eta L$ a $\sqrt{\eta L}$. Y el estocástico: el promedio implícito sobre pasos pasados actúa como filtro paso-bajo y reduce la varianza efectiva del gradiente sin sacrificar dirección útil.

Por qué $\gamma = 0.9$ es habitual. Con esa elección el peso de un gradiente recién calculado decae como una geométrica de razón $0.9$, así que la memoria efectiva es de $1/(1-\gamma) = 10$ pasos: $v_t$ es un promedio ponderado de los últimos diez gradientes con pesos $1, 0.9, 0.81, 0.73, \ldots$. Con $\gamma = 0.99$ la memoria sube a cien pasos, lo que da más suavidad pero más latencia cuando el gradiente cambia bruscamente.

Momentum es el origen conceptual del [[#Primer y segundo momento del gradiente|primer momento del gradiente]] de Adam y RMSProp: $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ es la misma idea reescrita como estimador estadístico de $\mathbb{E}[g]$ en lugar de "velocidad acumulada". La variante de Nesterov evalúa el gradiente en una posición predicha hacia adelante, $\nabla f(\theta - \gamma v_{t-1})$ en lugar de $\nabla f(\theta)$, lo que corrige por anticipado cuando el iterado está a punto de pasarse del mínimo.

**Papers:** [[An overview of gradient descent optimization algorithms|Ruder 2017]], [[Adam - A Method for Stochastic Optimization|Kingma & Ba 2015]], [[RMSProp - Divide the gradient by a running average of its recent magnitude|Tieleman & Hinton 2012]]

### Primer y segundo momento del gradiente

El primer momento es la media del gradiente $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, una media móvil exponencial (EMA) que estima $\mathbb{E}[g]$. El segundo momento no centrado es la media del cuadrado $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, una EMA que estima $\mathbb{E}[g^2]$. Adam mantiene ambos como su único estado por parámetro, y los valores por defecto $\beta_1 = 0.9$, $\beta_2 = 0.999$ definen las dos memorias efectivas: unas diez actualizaciones para $m$, unas mil para $v$.

Las dos cantidades responden a preguntas distintas. $m_t$ contesta "¿hacia dónde va consistentemente el gradiente?" y captura la señal direccional, en el mismo espíritu que [[#Momentum]]. $v_t$ contesta "¿cuán grande es el gradiente típicamente?" y captura la magnitud al cuadrado, que mezcla señal y varianza. La división $\hat m_t / (\sqrt{\hat v_t} + \varepsilon)$ las combina. La dirección viene del numerador filtrado de ruido, y la magnitud queda normalizada para que el paso efectivo no dependa de la escala absoluta del gradiente. Esto es lo que da pie a la [[#Tasa de aprendizaje adaptativa]] de Adam.

Falta un detalle de implementación, la corrección de sesgo. Como $m_t$ y $v_t$ se inicializan a cero, las EMAs arrancan sesgadas hacia cero. Tras un único paso, $m_1 = (1-\beta_1) g_1$ con $\beta_1 = 0.9$ es solo $0.1 \cdot g_1$. Si $g_t = 1$ es constante, sin corregir se tiene $m_1 = 0.1$, $m_2 = 0.19$, $m_3 = 0.271$, y hacen falta unas treinta iteraciones para acercarse a $1$; durante esa fase Adam daría pasos diez veces más pequeños de lo debido, y el segundo momento sin corregir daría un denominador minúsculo que dispararía el paso. La corrección $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$ lo deshace exactamente: en $t=1$ el factor vale $1/(1-0.9) = 10$, así que $\hat m_1 = g_1$ recupera el gradiente real, y a medida que $t$ crece $\beta^t \to 0$ y la corrección se desactiva.

**Papers:** [[Adam - A Method for Stochastic Optimization|Kingma & Ba 2015]], [[RMSProp - Divide the gradient by a running average of its recent magnitude|Tieleman & Hinton 2012]]

### Tasa de aprendizaje adaptativa

Una tasa de aprendizaje adaptativa por parámetro asigna a cada componente $\theta_j$ un learning rate efectivo $\eta_j$ que se ajusta a la geometría local del gradiente. Hace falta porque las capas y direcciones de una red tienen escalas y curvaturas muy distintas, y un único $\eta$ global obliga a comprometer: o se va demasiado lento donde el gradiente es pequeño (capas finales, embeddings raros) o se explota donde es grande. La receta general es dividir el paso global por una estimación de la magnitud reciente del gradiente, $\eta_{\text{eff}} = \eta / (\sqrt{v_j} + \varepsilon)$ con $v_j$ una acumulación de $g_j^2$, y las variantes se distinguen en cómo acumulan. AdaGrad suma todo el historial, lo que es robusto contra outliers tempranos pero decrece el paso monótonamente hasta detenerlo antes de tiempo. RMSProp y AdaDelta sustituyen la suma por una media móvil exponencial $v_j \leftarrow \rho v_j + (1-\rho) g_j^2$, con el mismo efecto adaptativo pero sin asfixiar el aprendizaje. Adam combina esa EMA con el [[#Primer y segundo momento del gradiente|primer momento]] tipo [[#Momentum]] y añade corrección de sesgo en los primeros pasos.

El valor de adaptar se ve en un word embedding con vocabulario grande. La palabra "el" aparece miles de veces por epoch y acumula gradientes enormes; una palabra rara como "criptografía" puede aparecer dos veces en todo el dataset. Con paso constante, "el" se mueve mucho en cada paso y "criptografía" apenas se actualiza. El denominador $\sqrt{v_j}$ es grande para "el" y pequeño para "criptografía", así que el paso efectivo se equilibra solo y cada palabra recibe un trato adaptado a su frecuencia.

Hay una lectura del ratio $\hat m / \sqrt{\hat v}$ que lo conecta con el [[#SNR del gradiente]]. No es solo una normalización por magnitud, es un SNR por parámetro, con la señal acumulada en el numerador y la magnitud típica (señal más ruido) en el denominador. Con alta certidumbre direccional el ratio se acerca a $1$ y Adam da un paso comparable al learning rate base; con gradientes per-sample que se cancelan, el numerador se hace pequeño y el paso se atenúa solo. Equivale a un schedule adaptativo sin programarlo, y explica por qué Adam suele ser robusto frente a la elección de $\eta$ inicial.

**Papers:** [[An overview of gradient descent optimization algorithms|Ruder 2017]], [[RMSProp - Divide the gradient by a running average of its recent magnitude|Tieleman & Hinton 2012]], [[Adam - A Method for Stochastic Optimization|Kingma & Ba 2015]]

### LR decay

En descenso por gradiente determinista (full-batch) sobre una función $L$-suave no hace falta decaer el learning rate. En SGD sí, porque el gradiente estimado $\hat{g}$ cumple $\mathbb{E}[\hat{g}] = \nabla f$ pero tiene varianza no nula, y esa varianza no desaparece en el óptimo. Incluso en $x^*$, donde $\nabla f = 0$, el gradiente estocástico conserva magnitud típica $\sigma > 0$ porque cada minibatch apunta a un sitio ligeramente distinto.

Con paso constante $\eta$, las iteradas no convergen a un punto. Se quedan rebotando en una vecindad del óptimo cuyo radio es proporcional a $\eta$. Formalmente, para un objetivo fuertemente convexo el suboptimum asintótico tiene un suelo lineal en $\eta$. Colapsar esa bola de ruido a un punto exige $\eta \to 0$. Ese es el argumento esencial: se decae el lr para apagar progresivamente el ruido del gradiente y poder converger al minimizador en lugar de orbitarlo.

**Papers:** [[A Study of Gradient Variance in Deep Learning|Faghri et al. 2020]], [[An overview of gradient descent optimization algorithms|Ruder 2017]], [[An Empirical Model of Large-Batch Training|McCandlish et al. 2018]]

---

## Generalización y features

### Sobreparametrización

Una red sobreparametrizada tiene muchos más parámetros que ejemplos de entrenamiento, lo que es la situación normal en deep learning moderno. Contra la intuición clásica de bias-variance, sobreparametrizar no degrada la generalización sino que la mejora hasta cierto punto, e incluso da pie al fenómeno de "double descent": añadir parámetros más allá del punto de interpolación (train loss exactamente cero) sigue bajando el test loss en lugar de subirlo.

La conexión con la alineación de gradientes va por la [[#Gradient confusion]]. Más anchura la reduce, porque hay tantas direcciones disponibles en el espacio de parámetros que dos ejemplos rara vez se ven obligados a competir por la misma, y eso permite entrenar con learning rate constante sin que SGD se estanque. Entrenando una red de dos capas en un problema sintético, la confusion empírica $\eta = -\min_{i\neq j}\cos(g_i, g_j)$ decae aproximadamente como $1/h$ con la anchura $h$ de la capa oculta: para $h = 10$ los gradientes per-sample chocan frontalmente y SGD necesita learning rate decreciente, y para $h = 1000$ están prácticamente desacoplados.

La profundidad hace lo contrario en redes vanilla, y durante años fue una barrera dura para pasar de unas decenas de capas. Lo que desbloqueó el problema fueron tres invenciones que reducen el efecto adverso de la profundidad sobre la confusion sin renunciar a ella como recurso expresivo: inicializaciones tipo Glorot, He y ortogonal, que mantienen estable la varianza de activaciones y gradientes a través de las capas; batch normalization; y skip connections, que dejan fluir el gradiente sin atravesar todas las capas no lineales. La combinación de las tres define la familia ResNet.

**Papers:** [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent|Sankararaman et al. 2020]]

### Feature learning

Feature learning es el fenómeno por el que las representaciones intermedias de una red evolucionan durante el entrenamiento y se alinean con la estructura del target. Operacionalmente, las capas inferiores aprenden funciones que vuelven más coherentes los gradientes vistos por las capas superiores: el [[#SNR del gradiente|GSNR]] sube, la [[#Stiffness]] intra-clase crece. Es lo que distingue una red profunda de un modelo de representación congelada en la inicialización, cuyo rendimiento está acotado por las features que se le den al inicio.

Un ejemplo que aclara la diferencia. Clasificando dígitos de MNIST, kernel ridge regression con un kernel gaussiano fijo representa cada imagen por sus distancias a un conjunto de "anchors" y solo aprende los pesos que las combinan; una CNN pequeña entrenada con SGD desarrolla filtros que detectan trazos curvos, esquinas y bucles, compuestos jerárquicamente capa a capa. Los dos alcanzan accuracy comparable en MNIST porque la tarea es fácil, pero en tareas más difíciles (CIFAR-100, ImageNet) la CNN sigue funcionando y el kernel no escala.

La firma medible es el signo de la trayectoria. En un modelo lineal, cantidades como el GSNR o la m-coherence decrecen monótonamente durante el entrenamiento, porque el gradiente residual contiene cada vez menos señal. En una red profunda real suben durante una fase considerable antes de saturar, y ese ascenso es el feature learning en marcha. Las capas inferiores alinean las representaciones internas con la estructura del target. La duración y la altura de ese ascenso correlacionan con el [[#Gap de generalización]] final.

**Papers:** [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]], [[Stiffness - A New Perspective on Generalization in Neural Networks|Fort et al. 2019]]

### Memorización vs generalización

La misma red, entrenada con la misma optimización, admite dos comportamientos opuestos. Con etiquetas reales ajusta el dataset y deja un [[#Gap de generalización]] pequeño, lo que significa que las features aprendidas se transfieren al test set. Con etiquetas aleatorias también lo ajusta (train accuracy del 100 por ciento) pero el gap explota, porque cada ejemplo se memoriza individualmente sin ningún feature compartido. Esto es la paradoja de Zhang y colaboradores en 2017.

La pista de por qué la red elige una cosa u otra está en la estructura de los gradientes, no en la capacidad. Con etiquetas reales, los [[#Gradientes per-sample]] de ejemplos de la misma clase están alineados. Todas las imágenes de gato empujan los filtros convolutivos hacia detectar texturas de pelo y formas de hocico, así que el minibatch refuerza esas direcciones compartidas. Con etiquetas aleatorias la regla que hay que aprender no es compartida sino una tabla de pares ejemplo-etiqueta arbitrarios, apenas hay pasos que beneficien a muchos ejemplos a la vez, y la red paga capacidad por ejemplo en lugar de compartir features.

De ahí sale un experimento canónico. Se inyecta una fracción $p$ de etiquetas aleatorias y se separa el dataset en `pristine` (etiqueta original) y `corrupt` (etiqueta permutada). En cada paso se mide la fracción de reducción de pérdida que viene de cada subgrupo: $f_t^p = \langle g_t, g_t^p\rangle / \|g_t\|^2$ y análogamente $f_t^c$, con $f_t^p + f_t^c = 1$. El resultado típico es que durante las primeras epochs $f_t^p \approx 0.95$ aunque el subgrupo pristine sea solo el 50 por ciento del dataset, es decir, la red optimiza casi exclusivamente las direcciones de los ejemplos limpios, y solo cuando esas direcciones se agotan empieza a memorizar los corruptos y $f_t^p$ baja. Aprender lo generalizable antes que lo idiosincrático es lo que justifica [[#Early stopping]] sin validation set: parando cuando $f_t^p$ deja de dominar, o cuando la m-coherence cae, o cuando el GSNR colapsa, se captura el modelo con lo máximo de transferible y sin lo intransferible.

**Papers:** [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Chatterjee 2020]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]], [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]]

### Gap de generalización

El gap de generalización es la diferencia $L_{\text{test}} - L_{\text{train}}$ entre la pérdida del modelo evaluada en el conjunto de entrenamiento y la evaluada en el de test (o su equivalente en error de clasificación). Es la cantidad que toda la familia de métricas train-time intenta acotar o predecir sin tener que tocar el test set.

La paradoja motivadora es la de Zhang y colaboradores en 2017. La misma red ajusta tanto etiquetas reales (gap pequeño) como etiquetas aleatorias (train accuracy del 100 por ciento y test accuracy de azar), así que la capacidad expresiva no explica por sí sola el gap (ver [[#Memorización vs generalización]]). La pista está en el proceso de entrenamiento. Las redes que generalizan acumulan gradientes coherentes durante el ajuste: m-coherence alta, [[#SNR del gradiente|GSNR]] alto por parámetro, [[#Stiffness]] positiva intra-clase, gradient disparity bajo entre minibatches. Las que memorizan tienen gradientes que se cancelan entre ejemplos: m-coherence cercana a 1, GSNR colapsado, stiffness cercana a cero o negativa.

Esto convierte el gap en una variable observable durante el entrenamiento, no solo en el post-mortem. En NAS clásico hay que entrenar a fondo cada arquitectura candidata hasta convergencia para saber cuál es la mejor, con un coste prohibitivo; con un [[#Proxy de generalización train-time]] como TSE-EMA o GWA basta entrenar cada candidata pocas epochs y leer la métrica para predecir el ranking final. El mismo argumento aplica a [[#Early stopping]] sin validation set y a la detección de muestras ruidosas durante el entrenamiento.

**Papers:** [[Coherent Gradients An Approach to Understanding Generalization in Gradient Descent-based Optimization|Chatterjee 2020]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]], [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]], [[Speedy Performance Estimation for Neural Architecture Search|Ru et al. 2021]]

### Early stopping

Early stopping es una técnica de regularización implícita que detiene la optimización cuando una métrica indicadora de generalización deja de mejorar durante un número fijo de epochs consecutivos (la patience $p$). La variante clásica monitoriza la pérdida de validación y se queda con los pesos del epoch con mejor val loss. Las variantes recientes prescinden del validation set usando proxies puramente del training set, lo que importa cuando los datos son escasos o caros de etiquetar.

Funciona porque SGD aprende lo generalizable antes que lo idiosincrático. En la fase temprana los [[#Gradientes per-sample]] están alineados (alta [[#Coherencia de gradientes]], alto [[#SNR del gradiente|GSNR]], [[#Stiffness]] positiva intra-clase) y la red ajusta features compartidas; en algún momento esas features se agotan y la red empieza a memorizar idiosincrasias, los gradientes per-sample se vuelven ortogonales y el test loss sube aunque el train loss siga bajando. Entrenando una ResNet-18 en CIFAR-10 con ruido de etiquetas del 20 por ciento, sin parar se llega al 100 por ciento de train accuracy y a alrededor del 75 por ciento de test tras unas 200 epochs; parando entre las epochs 40 y 60, justo antes de que la red memorice los ejemplos corruptos, el test queda del orden del 85 por ciento, diez puntos por encima.

Los proxies train-time más usados como criterio de parada vienen de los papers de alineación de gradientes. La gradient disparity, la distancia $\ell_2$ entre gradientes de dos minibatches del training set, se basa en una cota PAC-Bayes y empieza a subir justo cuando la red entra en régimen de memorización. La GWA agregada por epoch hace algo parecido con el coseno entre gradientes per-sample y pesos del clasificador, sin necesidad de muestrear dos batches. TSE-EMA acumula la pérdida de entrenamiento con una EMA y detecta cuándo se estanca.

**Papers:** [[Disparity Between Batches as a Signal for Early Stopping|Forouzesh & Thiran 2021]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]], [[Speedy Performance Estimation for Neural Architecture Search|Ru et al. 2021]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]]

### Proxy de generalización train-time

Un proxy de generalización train-time es una métrica calculada durante el propio entrenamiento, usando solo el training set, que predice el rendimiento futuro en el test set sin necesidad de un validation set separado. Su interés es triple: permite [[#Early stopping]] sin sacrificar datos a un split de validación, abarata drásticamente NAS al evitar entrenar a fondo cada candidato, y permite detectar muestras ruidosas o influyentes antes de tener un modelo final.

La familia se divide en dos grupos según qué observan. Las métricas de alineación miden cuánto comparten los ejemplos sus direcciones de gradiente: m-coherence, [[#Stiffness]], GWA, [[#SNR del gradiente|GSNR]]. Todas se construyen sobre [[#Gradientes per-sample]] y asumen que coherencia alta implica generalización. Las métricas de varianza o disparidad miran lo contrario: gradient disparity (distancia $\ell_2$ entre gradientes de dos minibatches), NGV (varianza normalizada del gradiente), TSE-EMA (acumulación EMA de pérdidas durante SGD). Los dos grupos correlacionan estrechamente, porque coherencia alta implica varianza y disparidad bajas, así que los rankings inducidos suelen estar de acuerdo.

El criterio de evaluación estándar es la correlación de Spearman o Kendall entre el ranking que induce el proxy y el ranking real de test. En NAS-Bench-201, con 15 625 arquitecturas candidatas, entrenarlas todas a fondo durante 200 epochs cuesta del orden de miles de horas de GPU; TSE-EMA medido tras solo 10 epochs alcanza una Kendall de aproximadamente 0.7 con el ranking real y a 25 epochs supera 0.8, suficiente para identificar el top-10 con una fracción del cómputo. Un proxy útil tiene que ser barato, robusto entre tareas y mejor que los zero-cost proxies, que se evalúan en la inicialización sin entrenar nada (JacCov, SNIP, SynFlow, NASWOT). Estos son aún más baratos, pero su correlación con el rendimiento real es inestable entre tareas.

**Papers:** [[Speedy Performance Estimation for Neural Architecture Search|Ru et al. 2021]], [[Disparity Between Batches as a Signal for Early Stopping|Forouzesh & Thiran 2021]], [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks|Hölzl 2025]], [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment|Chatterjee & Zielinski 2020]], [[Understanding Why Neural Networks Generalize Well Through GSNR of Parameters|Liu et al. 2020]]
