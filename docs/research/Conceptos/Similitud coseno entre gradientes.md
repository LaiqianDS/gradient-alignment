
El coseno entre dos gradientes $\cos(g_1, g_2) = \langle g_1, g_2\rangle / (\|g_1\|\,\|g_2\|)$ es la medida elemental de alineación direccional. Vale $1$ si los dos vectores apuntan en la misma dirección, $0$ si son perfectamente ortogonales, y $-1$ si son opuestos. Aparece en casi todas las métricas que cuantifican coherencia, conflicto o cooperación entre ejemplos, y es importante entender por qué se usa el coseno en lugar del producto interno crudo.

La razón es que el coseno normaliza por magnitud. Imagina dos ejemplos del dataset cuyos gradientes son $g_1 = (10, 0)$ y $g_2 = (1, 1)$. El producto interno es $10$, lo que parece indicar fuerte alineación, pero esa cifra viene casi enteramente de que $g_1$ es muy grande, no de que los dos vectores compartan dirección. El coseno da $\cos(g_1, g_2) = 10/(10\sqrt{2}) \approx 0.707$, que captura honestamente que el ángulo entre ellos es de 45 grados. Para coherencia y conflicto, esto es lo que importa, no la escala individual.

La misma operación, $\cos(\cdot, \cdot)$, aparece en al menos cuatro contextos distintos del campo de [[Gradientes per-sample]]. En [[Stiffness]] se calcula entre los gradientes de dos ejemplos individuales del training set, agregando luego por pares intra-clase y inter-clase. En GWA se calcula entre el gradiente per-sample y el vector de pesos del clasificador final, aprovechando la convergencia direccional de Ji y Telgarsky. En la variante mínima de [[Gradient confusion]] se mira el coseno más negativo entre cualquier par de gradientes para detectar el peor caso. Y en m-coherence el coseno aparece implícito al agregar $\|\sum g_i\|^2 / \sum \|g_i\|^2$, que se expande exactamente en una suma de cosenos por pares.

Una sutileza importante para el uso práctico. El coseno en alta dimensión tiene una distribución concentrada cerca de cero para vectores aleatorios: si los $g_i$ tienen $p$ componentes y son ruido isotrópico, el coseno típico entre dos cualesquiera es $O(1/\sqrt{p})$, casi exactamente cero. Eso significa que cualquier desviación sistemática positiva del coseno medio entre [[Gradientes per-sample]] es señal real, no fluctuación. En una red moderna con $p \sim 10^7$, observar cosenos promedio de $0.05$ ya es enorme, equivalente a varios cientos de desviaciones por encima de lo esperable por azar.

## Enlaces

- Stiffness sobre pares de ejemplos: [[Stiffness - A New Perspective on Generalization in Neural Networks]]
- GWA, coseno entre gradiente y pesos: [[Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks]]
- m-coherence como agregado de minibatch: [[Making Coherence Out of Nothing At All - Measuring the Evolution of Gradient Alignment]]
- Gradient confusion (mínimo coseno por pares): [[The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent]]
