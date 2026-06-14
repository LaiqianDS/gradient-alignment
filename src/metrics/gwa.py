"""Gradient-Weight Alignment (GWA) (Hölzl, 2025).

The per-sample score is the cosine similarity ``gamma(x_i) = cos(g_i, w_T)``
between the per-sample gradient and the final classifier weight vector ``w_T``.
The paper's convention is ``g = -∇L``; this pipeline operates on the RAW ∇L, so
every cosine sign is flipped relative to the paper — we keep the raw values
(ordering and magnitude are preserved) and do not negate (see ``compute``).

The per-epoch aggregate corrects for excess kurtosis,

    ``GWA_T = M1 / (M4/M2² - 3 + beta)``   with ``beta = 1.2``,

where ``M1`` is the mean cosine and ``M2``/``M4`` are the 2nd/4th central
moments; ``gwa/kurt`` logs the excess kurtosis ``M4/M2² - 3`` (Gaussian ≈ 0).
The canonical estimator is **last-layer-only**: the classifier weight matrix, bias
excluded (``docs/research/metrics.md``, "Gradient-Weight Alignment").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, iter_per_sample_grad_dicts, named_last_linear

# Excess-kurtosis correction constant from the paper (GWA_T denominator).
_BETA = 1.2


def _gwa_aggregate(gammas: torch.Tensor) -> dict[str, float]:
    """Kurtosis-corrected GWA aggregate over per-sample cosines ``gammas`` [M].

    Moments are accumulated in float64 (float32 underflows ``m2²`` once the
    cosine spread drops below ~1e-3): ``M1 = mean``, ``m2``/``m4`` the 2nd/4th
    central moments. ``gwa/kurt`` is the **excess** kurtosis ``m4/m2² - 3``
    (Gaussian ≈ 0), matching the paper's denominator. The aggregate
    ``value = M1 / (kurt + beta)`` guards its denominator with a
    sign-preserving ``EPS`` so a near-zero denominator stays finite without
    flipping the sign. Constant cosines (``m2 = 0``) leave the kurtosis
    undefined: ``gwa/kurt`` and ``gwa/value`` are NaN, ``gwa/score_mean`` stays.
    """
    if gammas.device.type == "mps":
        gammas = gammas.cpu()
    g = gammas.double()
    M1 = g.mean()
    m2 = ((g - M1) ** 2).mean()
    m4 = ((g - M1) ** 4).mean()
    if float(m2) == 0.0:
        return {
            "gwa/score_mean": float(M1),
            "gwa/kurt": float("nan"),
            "gwa/value": float("nan"),
        }
    kurt = m4 / m2**2 - 3.0  # excess kurtosis

    denom = kurt + _BETA
    # Sign-preserving guard: push the denominator away from zero in its own
    # direction (+EPS at exactly zero) so the ratio is finite and sign-stable.
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    value = M1 / (denom + sign * EPS)

    return {
        "gwa/score_mean": float(M1),
        "gwa/kurt": float(kurt),
        "gwa/value": float(value),
    }


class GwaMetric:
    """GWA metric: per-sample cosine to the classifier weights, then aggregate."""

    name = "gwa"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        lname, module = named_last_linear(model)
        w = module.weight.detach().reshape(-1)  # [W] classifier weight vector
        wn = w / w.norm().clamp_min(EPS)

        # cos(g_i, w) per sample, accumulated over streamed chunks (the head's
        # weight grads are tiny, so only the [M] cosines survive). We keep raw
        # ∇L: the paper's g = -∇L flips the sign, but negating w (or g) here would
        # only flip every gamma in unison, leaving order/magnitude unchanged.
        chunks = []
        for grads, _ in iter_per_sample_grad_dicts(model, X, y, loss_fn):
            g = grads[lname + ".weight"].flatten(start_dim=1)  # [c, W], bias excluded
            gn = g / g.norm(dim=1).clamp_min(EPS).unsqueeze(1)
            chunks.append(gn @ wn)  # [c]

        return _gwa_aggregate(torch.cat(chunks))


METRIC = GwaMetric()
