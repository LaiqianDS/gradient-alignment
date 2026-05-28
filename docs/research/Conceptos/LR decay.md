
En descenso por gradiente determinista (full-batch) sobre una función $L$-suave, no necesitas decaer el learning rate.

En SGD usas un gradiente estimado $\hat{g}$ con $\mathbb{E}[\hat{g}] = \nabla f$ pero varianza no nula. El problema es que esa varianza no desaparece en el óptimo: incluso en $x^*$, donde $\nabla f = 0$, el gradiente estocástico sigue teniendo magnitud típica $\sigma > 0$ porque cada minibatch apunta a un sitio ligeramente distinto.

Con paso constante $\eta$, las iteradas no convergen a un punto: se quedan rebotando en una vecindad del óptimo cuyo radio es proporcional a $\eta$. Formalmente, para un objetivo fuertemente convexo el suboptimum asintótico tiene un suelo lineal en $\eta$. Para colapsar esa bola de ruido a un punto necesitas $\eta \to 0$. Ese es el argumento esencial: decaes el lr para apagar progresivamente el ruido del gradiente y poder converger al minimizador en lugar de orbitarlo.

## Enlaces

- Eje varianza del gradiente: [[A Study of Gradient Variance in Deep Learning]]
- Schedules clásicos y motivación práctica: [[An overview of gradient descent optimization algorithms]]
- Relación batch size ↔ ruido ↔ paso: [[An Empirical Model of Large-Batch Training]]
