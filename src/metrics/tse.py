"""Training Speed Estimator (TSE) (Ru et al., 2021).

Consumes per-epoch mean training losses ℓ̄_1..ℓ̄_T, never a model, so its
``compute`` signature differs from the ``Metric`` protocol. ``t`` indexes
epochs: aggregate per-step losses into epoch means before calling.

Variants:
  * TSE      = Σ_t ℓ_t                        -> ``tse/cumulative``
  * TSE-E    = Σ_{t=T-E+1}^T ℓ_t (burn-in E)  -> ``tse/e_window``
  * TSE-EMA  = Σ_t γ^(T-t) ℓ_t, γ∈{0.9,0.999} -> ``tse/ema_0_9``, ``tse/ema_0_999``

In TSE-EMA the most recent loss (t=T) carries weight γ^0 = 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_tse(
    losses: Sequence[float] | torch.Tensor,
    *,
    e: int = 1,
    gammas: Sequence[float] = (0.9, 0.999),
) -> dict[str, float]:
    """TSE baseline scalars from a 1-D sequence of per-epoch mean losses ℓ̄_1..ℓ̄_T.

    Returns the cumulative sum, the burn-in window sum over the last ``e``
    losses (clamped to the available history when ``e > T``), and one EMA per
    γ in ``gammas`` keyed ``tse/ema_<g>`` (dot → underscore). An empty history
    returns all zeros.
    """
    L = torch.as_tensor(losses, dtype=torch.float64).flatten()
    T = L.shape[0]

    out: dict[str, float] = {
        "tse/cumulative": float(L.sum()),
        "tse/e_window": float(L[max(T - e, 0) :].sum()),
    }
    for g in gammas:
        exps = torch.arange(T - 1, -1, -1, dtype=torch.float64)
        weights = float(g) ** exps
        key = "tse/ema_" + str(g).replace(".", "_")
        out[key] = float((weights * L).sum())
    return out


METRIC = compute_tse
