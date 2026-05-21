# Métricas — Especificación de implementación

Síntesis de la revisión de los 15 papers de `Papers/` con la decisión cerrada de qué entra al `METRIC_REGISTRY` y cómo. Diez métricas en el registro (7 alineación + 3 varianza) más un predictor baseline (TSE-EMA). Lista cerrada antes de ejecutar experimentos (anti p-hacking).

Fuentes: `metrics.md` (resumen) + `Estado TFG.md` (diseño) + los 15 `.md` de papers.

---

## Convenciones del pipeline

Aplicables a todas las métricas:

- Calcular sobre el **gradiente bruto** $\nabla L(w)$, nunca sobre la actualización preacondicionada de Adam. Garantiza comparabilidad entre optimizadores.
- Durante la medición: snapshot de `model.state_dict()` + `optimizer.state_dict()`, `model.eval()` (congela BN/Dropout), restaurar al terminar. Crítico para Adam (m_t, v_t).
- Per-sample gradients vía `torch.func.vmap(torch.func.grad(loss_fn))` sobre un `probe_set` fijo (mismos índices en todas las épocas y runs de una configuración).
- Helper compartido `sample_batch_gradients(model, loader, K)` devuelve K gradientes de batch completos sin paso de optimizador.
- Cada métrica es una clase en `METRIC_REGISTRY` con `.cadence ∈ {"step", "epoch", "every_k_epochs"}` y `__call__(model, grads_dict)` que devuelve `float` o `dict[str, float]`.
- Almacenamiento: `experiments/raw/<run_id>/metrics.parquet`. Una fila por `(step_or_epoch, metric_name, value)`.

Tamaños por defecto propuestos (sujetos a ajuste tras pilot):

- `probe_size = 512` examples, estratificado por clase.
- `K_batches = 10` para métricas batch-grad.
- `s = 5` para `gradient_disparity` (definición del paper).
- `M_confusion = 50` (recortado del paper para que quepa en memoria).

---

## Familia 1 — Alineación / coherencia

### 1.1 `cos_sim_batches` — baseline básico

**Fórmula.** Cosine similarity media entre todos los pares de K gradientes de batch:
$$\text{cos\_sim\_batches} = \binom{K}{2}^{-1} \sum_{i<j} \frac{g_i \cdot g_j}{\|g_i\|\,\|g_j\|}.$$
Con $g_k \in \mathbb{R}^P$ el gradiente plano de un mini-batch en $w_t$.

**Inputs.** K gradientes de batch (mismos del helper compartido). Sin per-sample. K = 10.

**Coste.** $O(K \cdot \text{bwd}) + O(K^2 P)$. Memoria $K \cdot P$ floats (= 470MB para K=10, ResNet-18 fp32). Aceptable.

**Pitfalls.**
- Compartir los mismos K batches con `gradient_disparity` / `gradient_confusion` / `normalized_variance` (todos consumen el mismo barrido).
- `model.eval()` obligatorio; sin él, BN running stats divergen entre batches y se mide ruido.
- Para K pequeño la media puede estar dominada por un par outlier. Reportar también `median_cos` y `min_cos`.

**Sketch.**
```python
class CosSimBatches:
    cadence = "epoch"
    def __init__(self, K: int = 10):
        self.K = K

    def __call__(self, model, grads):
        gs = grads["batch_grads"]  # list of K flat tensors
        G = torch.stack([g / (g.norm() + 1e-12) for g in gs])  # (K, P)
        C = G @ G.T
        iu = torch.triu_indices(self.K, self.K, offset=1)
        sims = C[iu[0], iu[1]]
        return {
            "mean": sims.mean().item(),
            "median": sims.median().item(),
            "min": sims.min().item(),
        }
```

**Cadencia.** `epoch` (en `every_k_epochs=1` durante la ventana temprana 5–50%).

**Signo.** Mayor = mejor (gradientes alineados → SGD eficiente).

---

### 1.2 `gradient_disparity` — Forouzesh & Thiran (2021)

**Fórmula.** Distancia L2 media entre pares de s gradientes de batch:
$$\mathcal{D}_{i,j} = \|g_i - g_j\|_2, \qquad \overline{\mathcal{D}} = \binom{s}{2}^{-1} \sum_{i<j} \mathcal{D}_{i,j}.$$
Paper usa $s = 5$ (10 pares).

**Inputs.** s = 5 gradientes de batch. Comparte el barrido con `cos_sim_batches` si K ≥ s; en pilot mantenemos K = 10 y tomamos los primeros 5.

**Coste.** $O(s \cdot \text{bwd}) + O(s^2 P)$. Trivial cuando se comparte el barrido.

**Pitfalls.**
- Es magnitud L2, no coseno: contrae cuando $\|g\| \to 0$ tarde en entrenamiento. No normalizar; la teoría PAC-Bayes del paper depende de $\|g_1 - g_2\|^2$ sin normalizar.
- Los s batches deben ser **independientes**, no extracciones consecutivas del iterador que puedan solapar con el siguiente train step.

**Sketch.**
```python
class GradientDisparity:
    cadence = "epoch"
    def __init__(self, s: int = 5):
        self.s = s

    def __call__(self, model, grads):
        gs = grads["batch_grads"][: self.s]
        dists = [torch.linalg.norm(gs[i] - gs[j])
                 for i in range(self.s) for j in range(i + 1, self.s)]
        return torch.stack(dists).mean().item()
```

**Cadencia.** `epoch`.

**Signo.** Mayor = peor (correlación de Pearson 0.957 con test error en el paper).

---

### 1.3 `m_coherence` — Chatterjee & Zielinski (2020)

**Fórmula.** Con $g_i$ per-sample sobre probe set de tamaño $m$:
$$\alpha = \frac{\mathbb{E}_{z,z'}[g_z \cdot g_{z'}]}{\mathbb{E}_z[g_z \cdot g_z]} = \frac{\|\sum_i g_i\|^2}{m \sum_i \|g_i\|^2}, \qquad \alpha_m = m \cdot \alpha = \frac{\|\sum_i g_i\|^2}{\sum_i \|g_i\|^2}.$$
Rango: $\alpha_m \in [1, m]$. Igual a 1 cuando ortogonal; igual a $m$ cuando idéntico.

**Inputs.** Per-sample gradients sobre probe fijo $m \in [512, 2048]$.

**Coste.** $O(m \cdot \text{bwd vmap'd})$ time, $O(P)$ memoria streaming (acumuladores $S = \sum_i g_i$, $Q = \sum_i \|g_i\|^2$).

**Pitfalls.**
- Streaming obligatorio: no materializar matriz $(m, P)$. Para $m = 1024$, $P = 11.7M$ son ~47GB fp32.
- $\sum_i \|g_i\|^2$ underflow en fp16: forzar fp32.
- Mini-batches inflan coherencia (Cor 3.1 del paper). **Solo per-sample**, nunca per-batch.
- Diagonal $z = z'$ se mantiene en la definición.

**Sketch.**
```python
class MCoherence:
    cadence = "epoch"
    def __init__(self, probe_size: int = 1024, micro: int = 64):
        self.probe_size = probe_size
        self.micro = micro

    def __call__(self, model, grads):
        loader = grads["probe_loader"]   # yields (x, y) chunks of size `micro`
        device = grads["device"]
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def loss_one(p, b, x, y):
            out = torch.func.functional_call(model, (p, b), (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))
        grad_fn = torch.func.vmap(torch.func.grad(loss_one),
                                  in_dims=(None, None, 0, 0))

        S = None
        Q = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            gtree = grad_fn(params, buffers, xb, yb)
            g = torch.cat([t.reshape(t.shape[0], -1)
                           for t in gtree.values()], dim=1)  # (micro, P)
            S = g.sum(0) if S is None else S + g.sum(0)
            Q += (g * g).sum().item()
            n += g.shape[0]
        return (S.dot(S).item()) / max(Q, 1e-30)   # = α_m
```

**Cadencia.** `epoch` (o `every_k_epochs=2` si el pilot muestra coste prohibitivo).

**Signo.** Mayor = mejor (alta coherencia temprana → convergencia rápida y mejor generalización).

---

### 1.4 `gradient_confusion` — Sankararaman et al. (2020)

**Fórmula.** Mínimo coseno entre M gradientes de batch independientes:
$$\hat\eta = -\min_{i \neq j} \frac{g_i \cdot g_j}{\|g_i\|\,\|g_j\|}.$$
Paper recomienda $M = 100$ pares al final de cada época. Para nuestra escala recomendamos $M = 50$.

**Inputs.** M gradientes de batch, índices disjuntos.

**Coste.** $O(M \cdot \text{bwd}) + O(M^2 P)$ + memoria $M \cdot P$. Para $M = 50$, ResNet-18 fp32 ≈ 2.3GB — chunkear o pasar a CPU si peta.

**Pitfalls.**
- Min sobre $\binom{M}{2}$ es estimador de extremo, ruidoso. Reportar también `median`, `p05`, `frac_neg` por estabilidad.
- Compartir batches con `cos_sim_batches`/`normalized_variance` si $K = M$; si no, barrido independiente.
- Batches deben ser disjuntos en una medición; muestrear sin reemplazo.

**Sketch.**
```python
class GradientConfusion:
    cadence = "epoch"
    def __init__(self, M: int = 50):
        self.M = M

    def __call__(self, model, grads):
        gs = grads["batch_grads"][: self.M]
        G = torch.stack([g / (g.norm() + 1e-12) for g in gs])  # (M, P)
        C = G @ G.T
        iu = torch.triu_indices(self.M, self.M, offset=1)
        sims = C[iu[0], iu[1]]
        return {
            "eta_hat": (-sims.min()).item(),
            "min_cos": sims.min().item(),
            "median_cos": sims.median().item(),
            "p05_cos": sims.quantile(0.05).item(),
            "frac_neg": (sims < 0).float().mean().item(),
        }
```

**Cadencia.** `epoch`.

**Signo.** `eta_hat` mayor = peor. `min_cos` mayor = mejor (preferible para consistencia con otras métricas coseno).

---

### 1.5 `stiffness` — Fort et al. (2019)

**Fórmula.** Para gradientes per-sample $g_i = \nabla_W \ell(f_W(x_i), y_i)$ sobre probe estratificado de tamaño $M$:

- Cosine-stiffness: $S_{\cos}(i,j) = (g_i \cdot g_j)/(\|g_i\|\,\|g_j\|)$
- Sign-stiffness: $S_{\text{sign}}(i,j) = \text{sign}(g_i \cdot g_j)$
- Agregados: $\bar S_{\cos} = \mathbb{E}_{i \neq j}[S_{\cos}(i,j)]$, $\bar S_{\cos}^{\text{within}}$ (mismo label), $\bar S_{\cos}^{\text{between}}$ (distinto label).

**Inputs.** Per-sample grads sobre probe $M = 256$ estratificado + labels.

**Coste.** $O(M \cdot \text{bwd vmap'd}) + O(M^2 P)$ con gram matriz $G G^\top$. Memoria $M \cdot P$ floats = **12GB para $M = 256$, ResNet-18 fp32**. Bloqueante sin mitigación.

**Pitfalls.**
- Memoria $M \cdot P$ es el cuello. Soluciones: chunk en M (acumular filas de la gram), last-layer-only, JL projection.
- $\|g_i\| = 0$ raro pero posible; epsilon en denominador, o máscara.
- vmap dobla la memoria de activación; ajustar `chunk_size`.

**Mitigación recomendada.**
- **Default last-layer only** en ResNet-18 ($P_{\text{eff}} \approx 5k$ en lugar de 11.7M). Consistente con GWA. Stiffness por capa sigue siendo teóricamente significativo (Fort lo reporta por bloques).
- En FC/CNN-small: full-parameter posible (P pequeño).

**Sketch.**
```python
class Stiffness:
    cadence = "epoch"
    def __init__(self, probe_size: int = 256, chunk: int = 32, scope: str = "last_layer"):
        self.probe_size = probe_size
        self.chunk = chunk
        self.scope = scope

    def __call__(self, model, grads):
        probe_x = grads["probe_x"]
        probe_y = grads["probe_y"]
        params = {k: v.detach() for k, v in model.named_parameters()
                  if self._in_scope(k)}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def loss_one(p, b, x, y):
            out = torch.func.functional_call(model, (p, b), (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))
        per_sample = torch.func.vmap(torch.func.grad(loss_one),
                                     in_dims=(None, None, 0, 0))

        rows = []
        for xb, yb in zip(probe_x.split(self.chunk), probe_y.split(self.chunk)):
            gtree = per_sample(params, buffers, xb, yb)
            flat = torch.cat([v.reshape(v.shape[0], -1)
                              for v in gtree.values()], dim=1)
            rows.append(flat / (flat.norm(dim=1, keepdim=True) + 1e-12))
        G = torch.cat(rows, dim=0)        # (M, P_eff) unit norm
        C = G @ G.T                        # (M, M) cosine
        mask = ~torch.eye(C.size(0), dtype=torch.bool, device=C.device)
        same = (probe_y[:, None] == probe_y[None, :]) & mask
        diff = (probe_y[:, None] != probe_y[None, :]) & mask
        return {
            "cos": C[mask].mean().item(),
            "sign": C.sign()[mask].mean().item(),
            "cos_within": C[same].mean().item(),
            "cos_between": C[diff].mean().item(),
        }

    def _in_scope(self, name: str) -> bool:
        if self.scope == "last_layer":
            return name.startswith("fc.") or name.startswith("classifier.")
        return True
```

**Cadencia.** `epoch` (o `every_k_epochs=2` en arquitecturas grandes).

**Signo.** `cos_within` mayor = mejor (clases con features compartidos). `cos_between` cerca de 0 esperable.

---

### 1.6 `gwa` — Hölzl (2025)

**Fórmula.** Score per-sample coseno entre gradiente per-sample y pesos del clasificador final:
$$\gamma(x_i, w_T) = \cos\text{sim}(g_T(x_i), w_T), \qquad g_T(x_i) = -\nabla_w \mathcal{L}(w_T, x_i).$$
GWA agregado con corrección de curtosis:
$$\text{GWA}_T = \frac{M^{(1)}}{M^{(4)}/(M^{(2)})^2 - 3 + \beta}, \qquad \beta = 1.2,$$
donde $M^{(1)}$ es la media (momento no central) y $M^{(2..4)}$ son momentos centrales. Estimador cerrado para la última capa lineal:
$$g_t(x_i) = -z_i (\hat y_i - y_i)^\top,$$
con $z_i$ latente penúltima, $\hat y_i = \text{softmax}(\text{logits})$.

**Inputs.** Latentes $z_i$ + softmax probs + targets + $w_T$ (matriz del clasificador). Online sobre los mini-batches de entrenamiento de cada época.

**Coste.** **Casi gratis**: $O(B \cdot d_z \cdot C)$ por mini-batch, despreciable frente a forward+backward. Sin retroceso adicional.

**Pitfalls.**
- Excluir el bias del clasificador (norma cero → coseno indefinido).
- Welford estable para $M^{(2..4)}$. Computar curtosis desde $\sum \gamma^k$ crudos es numéricamente malo.
- $\beta = 1.2$ mantiene denominador positivo en límite uniforme; añadir clamp $\max(\cdot, \epsilon)$ como salvavida.
- Convención de signo: paper define $g = -\nabla L$; en pipeline usamos $\nabla L$ bruto. O guardamos $\gamma_i$ con signo del paper, o reportamos $-\cos$ y documentamos.
- Para arquitectura FC: el clasificador final **es** la red completa. Caso degenerado, flag explícito.

**Sketch.**
```python
class GWA:
    cadence = "epoch"
    def __init__(self, beta: float = 1.2, eps: float = 1e-30):
        self.beta = beta
        self.eps = eps
        self._reset()

    def _reset(self):
        self.n = 0
        self.m1 = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0

    def update(self, z, logits, y, classifier_W):
        # Called from training loop after forward, before backward.
        # z: (B, d_z); logits: (B, C); y: (B,); classifier_W: (C, d_z)
        with torch.no_grad():
            p = logits.softmax(-1)
            err = p - F.one_hot(y, p.size(-1)).float()        # (B, C)
            g = -torch.einsum("bc,bd->bcd", err, z).flatten(1) # (B, C*d_z)
            w = classifier_W.flatten()
            gamma = F.cosine_similarity(g, w.unsqueeze(0), dim=1)
            # Welford running moments
            for x in gamma.cpu().tolist():
                self.n += 1
                delta = x - self.m1
                delta_n = delta / self.n
                term = delta * delta_n * (self.n - 1)
                self.m1 += delta_n
                self.M4 += (term * delta_n ** 2 * (self.n ** 2 - 3 * self.n + 3)
                            + 6 * delta_n ** 2 * self.M2
                            - 4 * delta_n * self.M3)
                self.M3 += (term * delta_n * (self.n - 2)
                            - 3 * delta_n * self.M2)
                self.M2 += term

    def __call__(self, model, grads):
        n = max(self.n, 1)
        var = self.M2 / n
        kurt = (self.M4 / n) / max(var ** 2, self.eps) - 3.0
        gwa = self.m1 / max(kurt + self.beta, self.eps)
        self._reset()
        return float(gwa)
```

**Cadencia.** `epoch` (acumulación online, scalar al cerrar la época).

**Signo.** Mayor = mejor (paper reporta Pearson 0.99 con test accuracy).

---

### 1.7 `ntk_alignment` — Shan & Bordelon (2021)

**Fórmula.** Kernel-Target Alignment (Cortes et al. 2012) sobre el NTK empírico:
$$\text{KTA} = \frac{\langle K, Y \rangle_F}{\|K\|_F \cdot \|Y\|_F}, \qquad K_{ij} = \nabla_\theta f(x_i) \cdot \nabla_\theta f(x_j).$$
Reducción escalar: $f(x_i) = z_{y_i}(x_i)$ (logit de la clase correcta). $Y$ = matriz one-vs-rest $\pm 1$ por pares: $Y_{ij} = +1$ si $y_i = y_j$, $-1$ si no.

**Inputs.** Per-sample **Jacobian** (no gradiente de loss) de $f$ vía `torch.func.vmap(torch.func.jacrev(f))`. Probe $N = 256$ + labels.

**Coste.** $O(N \cdot \text{bwd-jac})$ + memoria $N \cdot P$ (igual que stiffness — 12GB ResNet-18 sin mitigación). Importante: **barrido separado** del de stiffness porque calcula gradiente de $f$, no de $\ell$.

**Pitfalls.**
- $K = J J^\top$ es PSD por construcción; aproximaciones por random projection pueden romperlo.
- KTA con $Y$ uno-vs-rest $\pm 1$, no one-hot (que da rango trivial $\in [1/C, 1]$).
- Forma sin centrar (Shan & Bordelon la usan así; Cortes original la centra).
- Para KSM multiclase: $C \times$ el coste (un Jacobian por dimensión de output). Off-by-default; opcional cada 5 épocas en CIFAR-10.

**Mitigación recomendada.**
- **Last-layer-only Jacobian** como default. Theory en Shan & Bordelon es aditiva por capa.
- Reducción escalar (logit clase correcta), no Jacobian completo.

**Sketch.**
```python
class NTKAlignment:
    cadence = "epoch"
    def __init__(self, probe_size: int = 256, chunk: int = 32, scope: str = "last_layer"):
        self.probe_size = probe_size
        self.chunk = chunk
        self.scope = scope

    def __call__(self, model, grads):
        probe_x = grads["probe_x"]
        probe_y = grads["probe_y"]
        params = {k: v.detach() for k, v in model.named_parameters()
                  if self._in_scope(k)}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def f_correct(p, b, x, y):
            out = torch.func.functional_call(model, (p, b), (x.unsqueeze(0),))
            return out.squeeze(0)[y]

        per_sample_jac = torch.func.vmap(
            torch.func.jacrev(f_correct), in_dims=(None, None, 0, 0))

        rows = []
        for xb, yb in zip(probe_x.split(self.chunk), probe_y.split(self.chunk)):
            jtree = per_sample_jac(params, buffers, xb, yb)
            rows.append(torch.cat([v.reshape(v.shape[0], -1)
                                   for v in jtree.values()], dim=1))
        J = torch.cat(rows, dim=0)           # (N, P)
        K = J @ J.T                           # (N, N) PSD
        Y = (probe_y[:, None] == probe_y[None, :]).float() * 2 - 1
        kta = (K * Y).sum() / (K.norm() * Y.norm() + 1e-12)
        return {
            "kta": kta.item(),
            "K_fro": K.norm().item(),
            "K_trace": K.diag().sum().item(),
        }

    def _in_scope(self, name: str) -> bool:
        if self.scope == "last_layer":
            return name.startswith("fc.") or name.startswith("classifier.")
        return True
```

**Cadencia.** `epoch` (NTK cambia lento en régimen no-lazy).

**Signo.** Mayor = mejor (teorema 4.1 del paper: KTA alto acelera aprendizaje).

---

## Familia 2 — Varianza estocástica

### 2.1 `normalized_variance` — Faghri et al. (2020)

**Fórmula.**
$$\text{NGV}(w_t) = \frac{\text{tr}(\text{Cov}(g))}{\|\mathbb{E}[g]\|^2} = \frac{\frac{1}{K-1}(\sum_k \|g_k\|^2 - K \|\bar g\|^2)}{\|\bar g\|^2}.$$
$g_k$ gradiente de batch completo en $w_t$, $\bar g = \frac{1}{K} \sum_k g_k$.

**Inputs.** $K = 20$–$40$ gradientes de batch. Streaming accumulators $S = \sum_k g_k$, $Q = \sum_k \|g_k\|^2$.

**Coste.** $O(K \cdot \text{bwd})$, $O(P)$ memoria. Cheap.

**Pitfalls.**
- Estimador unbiased divide por $K - 1$.
- Cuando $\|\bar g\| \to 0$ el ratio explota. Añadir eps; loguear num/den por separado para auditar.
- NGV no es monótono; crece durante entrenamiento en CIFAR-10/ImageNet.
- NGV por capa puede divergir del global; loguear ambos opcional.

**Sketch.**
```python
class NormalizedVariance:
    cadence = "epoch"
    def __init__(self, K: int = 20, eps: float = 1e-12):
        self.K = K
        self.eps = eps

    def __call__(self, model, grads):
        gs = grads["batch_grads"]
        K = len(gs)
        S = torch.zeros_like(gs[0])
        Q = 0.0
        for g in gs:
            S += g
            Q += g.pow(2).sum().item()
        mean = S / K
        mean_sq_norm = mean.pow(2).sum().item()
        trace_cov = (Q - K * mean_sq_norm) / (K - 1)
        return trace_cov / (mean_sq_norm + self.eps)
```

**Cadencia.** `epoch`.

**Signo.** Menor = mejor (NGV < 1 → señal domina ruido).

---

### 2.2 `gns_simple` — McCandlish et al. (2018)

**Fórmula.** Versión simple (sin Hessiana):
$$\mathcal{B}_{\text{simple}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}.$$
Estimador de dos batches disjuntos $B_{\text{small}}, B_{\text{big}}$:
$$\hat{\|G\|^2} = \frac{B_{\text{big}}\|G_{\text{big}}\|^2 - B_{\text{small}}\|G_{\text{small}}\|^2}{B_{\text{big}} - B_{\text{small}}},$$
$$\hat{\text{tr}(\Sigma)} = \frac{\|G_{\text{small}}\|^2 - \|G_{\text{big}}\|^2}{1/B_{\text{small}} - 1/B_{\text{big}}}.$$

**Inputs.** Dos batches: $B_{\text{small}} = B_{\text{train}}$, $B_{\text{big}} = 4 B_{\text{train}}$, índices disjuntos.

**Coste.** $O(2 \cdot \text{bwd})$ — el batch big es ~4× el normal, así que ~5× un paso de entrenamiento.

**Pitfalls.**
- **No implementar `gns_exact`** (Hessian-weighted). HVPs son caros y `gns_simple` es suficiente per paper.
- $\mathcal{B}_{\text{simple}}$ crece 1–2 órdenes durante entrenamiento. Loguear en log-scale.
- Estimador puede dar $\hat{\text{tr}(\Sigma)}$ negativo cuando ruido domina señal: EMA opcional sobre num/den por separado antes de dividir.
- Compartir un único barrido con `normalized_variance` si ambos en epoch: el batch big puede ser concatenación de batches small.

**Sketch.**
```python
class GNSSimple:
    cadence = "epoch"
    def __init__(self, B_big_mult: int = 4, eps: float = 1e-12):
        self.B_big_mult = B_big_mult
        self.eps = eps

    def __call__(self, model, grads):
        g_small = grads["g_small"]  # flat tensor on a single train batch
        g_big = grads["g_big"]      # flat tensor on B_big_mult * train_batch
        B_s = grads["B_small"]
        B_b = grads["B_big"]
        n_s = g_small.pow(2).sum().item()
        n_b = g_big.pow(2).sum().item()
        G_est = (B_b * n_b - B_s * n_s) / (B_b - B_s)
        S_est = (n_s - n_b) / (1.0 / B_s - 1.0 / B_b)
        return S_est / (G_est + self.eps)
```

**Cadencia.** `epoch`.

**Signo.** Menor = menor ruido relativo. Relación con eficiencia más sutil que NGV (McCandlish vincula con batch size crítico, no directamente con épocas-a-umbral).

---

### 2.3 `gsnr` — Liu et al. (2020)

**Fórmula.** SNR por parámetro:
$$r(\theta_j) = \frac{\tilde g(\theta_j)^2}{\rho^2(\theta_j)}, \qquad \tilde g(\theta_j) = \mathbb{E}_i[g_{i,j}], \quad \rho^2(\theta_j) = \text{Var}_i[g_{i,j}].$$
Agregado: $\text{GSNR}_{\text{global}} = \text{mean}_j \, r(\theta_j)$. Per-layer opcional.

**Inputs.** Per-sample grads sobre probe fijo $M = 512$. Acumuladores $S_j = \sum_i g_{i,j}$, $Q_j = \sum_i g_{i,j}^2$.

**Coste.** $O(M)$ vmap, $O(P)$ memoria streaming. Misma sweep que `m_coherence` y `stiffness` cuando comparten probe set.

**Pitfalls.**
- Agregar con **mean** sobre parámetros, no sum (sum incomparable entre arquitecturas).
- $\rho_j^2$ unbiased: $\frac{1}{M-1} \sum_i (g_{i,j} - \bar g_j)^2$.
- Para parámetros con $\rho_j^2 \to 0$ (dead ReLU, capas congeladas), añadir eps por param. Considerar **median** como robusto.
- Cola pesada de la distribución de $r_j$: loguear también median + p95.
- **No implementar OSGR**: requiere M runs de entrenamiento independientes, incompatible con un run por composición.

**Sketch.**
```python
class GSNR:
    cadence = "every_k_epochs"
    every_k = 1
    def __init__(self, probe_size: int = 512, eps: float = 1e-12):
        self.probe_size = probe_size
        self.eps = eps

    def __call__(self, model, grads):
        probe_x = grads["probe_x"]
        probe_y = grads["probe_y"]
        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}

        def loss_one(p, b, x, y):
            out = torch.func.functional_call(model, (p, b), (x.unsqueeze(0),))
            return F.cross_entropy(out, y.unsqueeze(0))

        per_sample = torch.func.vmap(torch.func.grad(loss_one),
                                     in_dims=(None, None, 0, 0))
        gtree = per_sample(params, buffers, probe_x, probe_y)

        r_per_layer = {}
        r_all = []
        for k, g in gtree.items():
            g_flat = g.reshape(g.size(0), -1)         # (M, P_k)
            mean = g_flat.mean(dim=0)
            var = g_flat.var(dim=0, unbiased=True)
            r_k = mean.pow(2) / (var + self.eps)
            r_per_layer[k] = r_k.mean().item()
            r_all.append(r_k)
        r_cat = torch.cat(r_all)
        return {
            "mean": r_cat.mean().item(),
            "median": r_cat.median().item(),
            "p95": r_cat.quantile(0.95).item(),
            "per_layer": r_per_layer,
        }
```

**Cadencia.** `every_k_epochs=1`.

**Signo.** Mayor = mejor generalización (Liu: GSNR alto → gap pequeño).

---

## Baseline predictor — `tse_ema` (Ru et al. 2021)

**No entra en `METRIC_REGISTRY`.** Es un predictor de eficiencia, no una métrica de gradiente. Tagear `metric_kind="baseline"` en `metrics_at_window.parquet` para separarlo de las métricas en el análisis.

**Fórmula.**
$$\text{TSE} = \sum_{t=1}^{T} \bar\ell_t, \qquad \text{TSE-E} = \sum_{t=T-E+1}^{T} \bar\ell_t, \qquad \text{TSE-EMA} = \sum_{t=1}^{T} \gamma^{T-t} \bar\ell_t,$$
con $\bar\ell_t$ = train loss medio del step/época $t$ y $\gamma = 0.999$.

**Justificación.**
- Coste cero — el loss ya se computa.
- Sin él, la tesis no puede defender que las métricas de gradiente valen la instrumentación.
- Ru et al. muestran que TSE-EMA supera validation accuracy, learning-curve extrapolation, y zero-cost proxies en Spearman ρ.

**Sketch (en el training loop).**
```python
tse = 0.0
tse_ema = 0.0
GAMMA = 0.999

for epoch in range(num_epochs):
    epoch_loss = train_one_epoch(model, loader)
    tse += epoch_loss
    tse_ema = GAMMA * tse_ema + epoch_loss
    # snapshot at the 4 windows (5%, 10%, 25%, 50% of epochs)
```

Registrar `tse`, `tse_e` (E=1), `tse_ema` (γ=0.999) en cada ventana.

---

## Tabla resumen

| Métrica | Familia | Probe | K batches | Cadencia | Coste relativo | Signo (↑ = mejor) | Comparte sweep |
|---|---|---|---|---|---|---|---|
| `cos_sim_batches` | alineación | — | 10 | epoch | bajo | ↑ | batch-grad |
| `gradient_disparity` | alineación | — | 5 | epoch | bajo | ↓ | batch-grad |
| `m_coherence` | alineación | 1024 | — | epoch | medio | ↑ | per-sample (∇L) |
| `gradient_confusion` | alineación | — | 50 | epoch | medio-alto | ↑ (`min_cos`) | batch-grad |
| `stiffness` | alineación | 256 | — | epoch | alto | ↑ within | per-sample (∇L) |
| `gwa` | alineación | — | — | epoch | ~0 | ↑ | online en train fwd |
| `ntk_alignment` | alineación | 256 | — | epoch | alto | ↑ | per-sample (∇f) ✱ |
| `normalized_variance` | varianza | — | 20–40 | epoch | bajo | ↓ | batch-grad |
| `gns_simple` | varianza | — | 2 (s+b) | epoch | bajo | ↓ | batch-grad |
| `gsnr` | varianza | 512 | — | every_k_ep=1 | alto | ↑ | per-sample (∇L) |
| `tse_ema` | baseline | — | — | epoch | 0 | ↓ | online en train |

✱ NTK no comparte gradientes con stiffness/gsnr/m_coherence porque calcula Jacobian de $f$, no ∇L.

---

## Sweeps compartidos (clave para reducir coste)

Tres barridos cubren las 10 métricas:

1. **Batch-grad sweep** (K = 10 batches disjuntos, full-batch grads, no optimizer step):
   - Sirve: `cos_sim_batches`, `gradient_disparity` (s=5 primeros), `gradient_confusion` (M=10 o ampliar a 50 con extra), `normalized_variance` (K=10 o ampliar a 20)
   - Coste: 10 × bwd
   - Si `gradient_confusion` o `normalized_variance` necesitan M > 10, ampliar este sweep — son las dos que más demandan.

2. **Per-sample ∇L sweep** (probe fijo M = 1024, vmap'd):
   - Sirve: `m_coherence`, `stiffness` (con M = 256 ⊂ 1024 o sweep propio si M difiere), `gsnr` (M = 512)
   - Coste: ~1024 / chunk × bwd
   - Streaming de S y Q para `m_coherence`, gram-chunked para `stiffness`, per-param accum para `gsnr`. Tres outputs de una sola vmap.

3. **NTK Jacobian sweep** (probe fijo N = 256, vmap+jacrev):
   - Sirve: `ntk_alignment` (last-layer default)
   - Coste: ~256 / chunk × jac
   - Separado porque calcula ∇f, no ∇L.

`gwa` no requiere sweep — engancha hooks al forward de training para capturar latentes + softmax + classifier weights, acumula online.

---

## Pitfalls cross-cutting

- **Memoria $M \cdot P$**: bloqueante en ResNet-18 para `stiffness`, `gradient_confusion`, `ntk_alignment`. Default: last-layer-only en ResNet-18; full-param en FC y CNN-small.
- **fp16/fp32**: per-sample dot products y sumas de cuadrados pueden underflow en fp16. Forzar fp32 en agregados de métricas.
- **`model.eval()` durante medición**: obligatorio. BN running stats y Dropout introducen ruido espurio si no.
- **Snapshot/restore de optimizador**: crítico para Adam ($m_t$, $v_t$ acumulan entre steps). Sin restore, el run divergiría tras cada medición.
- **Convención de signo**: GWA usa $g = -\nabla L$ en el paper; el resto usa $\nabla L$. Guardar valores crudos y normalizar en la capa de análisis (no en el registro).
- **Determinismo**: `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`. Probe sets y batches del sweep semillados.

---

## Redundancias esperadas (validar en pilot)

- `normalized_variance` y `gns_simple` relacionados por CLT: $\mathcal{B}_{\text{simple}} \approx B \cdot \text{NGV}$. Spearman esperado > 0.9. Mantener ambos hasta confirmar.
- `cos_sim_batches.mean` vs `gradient_disparity`: si los $\|g_i\|$ son similares entre batches (caso típico), L2-disparity y 1-coseno son aproximadamente monótonos. Spearman esperado alto pero no perfecto.
- `gsnr.mean` vs `normalized_variance`: GSNR agrega por parámetro luego promedia; NGV agrega antes de dividir. Por Jensen son distintas. Spearman moderado esperado (~0.6–0.8).

Si pilot confirma redundancias, dropear la versión más cara antes del experimento principal.

---

## Métricas no implementadas (revisadas, descartadas)

- **OSGR** (Liu 2020): requiere M runs de entrenamiento independientes. Incompatible con un run por composición. GSNR es el candidato del registro.
- **KSM completo** (Shan & Bordelon): coste $C \times$ NTK escalar. Off-by-default; opcional `every_k_epochs=5` en CIFAR-10.
- **$f_t^p, f_t^c$** (Chatterjee 2019 Coherent Gradients): requiere label noise para definir pristine/corrupt. Label noise descartado en v1 del TFG.
- **`gns_exact`** (McCandlish con Hessiana): HVPs caros, paper muestra que `gns_simple` aproxima bien.
- **Stiffness vs distancia** (Fort): añade dimensión continua "distancia en input"; análisis 2D fuera de scope correlacional.

---

## Decisiones abiertas (validar antes de pilot)

1. **Probe size**: M = 512 (default) vs M = 1024 (más robusto, ~2× coste en per-sample sweep).
2. **Scope de stiffness / ntk_alignment en ResNet-18**: last-layer-only (default recomendado) vs full-param con chunking.
3. **K para batch-grad sweep**: K = 10 (cheap) vs K = 50 (mejor `gradient_confusion`).
4. **`gradient_confusion`**: M ⊂ K (compartido) vs M = 50 independiente.
5. **Per-layer logging**: además del scalar global, loguear per-layer en `m_coherence`, `gsnr`, `stiffness`, `normalized_variance`? Diagnóstico vs volumen de storage.
6. **Cadencia de pilot**: `epoch` para todo en v1; subir a `every_k_steps` solo en la ventana 5% si el pilot lo permite.

---

## Bibliografía citable (extraída de los 4 papers de background)

- **Chatterjee 2019** — CGH framework, motivación de toda la familia alineación. Intro + related work.
- **Ru et al. 2021** — TSE-EMA como baseline obligatorio. Methods + discussion.
- **Johnson & Zhang 2013** — SVRG, justificación teórica del eje varianza. Related work.
- **Kingma & Ba 2015 / Tieleman & Hinton 2012** — $\hat m_t / \sqrt{\hat v_t}$ como SNR, motiva NGV/GSNR. Related work + methods (raw-grad rationale).
- **Ruder 2017** — taxonomía optimizadores, citación de respaldo para sweep SGD+Adam.

Notas: Ru et al. + Hölzl 2025 + Forouzesh 2021 son los tres papers más comparables (mismo problema: predicción temprana de eficiencia/generalización con métrica barata). La tesis debe argumentar el delta vs estos tres en la intro.
