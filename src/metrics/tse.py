"""Training Speed Estimator (TSE), the mandatory baseline (Ru et al., 2021).

Unlike every gradient metric in the registry, this baseline consumes a
*sequence of per-epoch mean training losses* ℓ̄_1..ℓ̄_T — never a model — so its
``compute`` signature intentionally differs from the ``Metric`` protocol in
``metrics.base``. The paper defines t over **epochs** (Eq. 1: each term is the
mean batch loss of one epoch); callers must aggregate per-step losses first
(``train.epoch_mean_losses``). Cost is **zero**: the train loss is already
produced by the forward pass.

Variants (see ``docs/research/metrics.md``, tag ``metric_kind="baseline"``):
  * TSE      = Σ_t ℓ_t                       — ``tse/cumulative``
  * TSE-E    = Σ_{t=T-E+1}^T ℓ_t (burn-in E) — ``tse/e_window``
  * TSE-EMA  = Σ_t γ^(T-t) ℓ_t, γ∈{0.9,0.999} — ``tse/ema_0_9``, ``tse/ema_0_999``

In TSE-EMA the most recent loss (t=T) carries weight γ^0 = 1 and earlier
losses decay geometrically.
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
        # weight γ^(T-t): exponents T-1, T-2, ..., 0 so the last loss weighs 1
        exps = torch.arange(T - 1, -1, -1, dtype=torch.float64)
        weights = float(g) ** exps
        key = "tse/ema_" + str(g).replace(".", "_")
        out[key] = float((weights * L).sum())
    return out


class TseMetric:
    """Baseline predictor. NOTE: unlike gradient metrics, ``compute`` takes a
    loss sequence (ℓ_1..ℓ_T), not a model + data probe."""

    name = "tse"

    def compute(
        self,
        losses: Sequence[float] | torch.Tensor,
        *,
        e: int = 1,
        gammas: Sequence[float] = (0.9, 0.999),
    ) -> dict[str, float]:
        return compute_tse(losses, e=e, gammas=gammas)


METRIC = TseMetric()
