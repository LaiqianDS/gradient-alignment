"""Gradient noise scale, simple per-sample estimator (McCandlish et al., 2018).

Implements the simplified form under :math:`H \\propto I`,
:math:`\\mathcal{B}_{\\text{simple}} = \\mathrm{tr}(\\Sigma)/\\|G\\|^2`, estimated
per-sample as :math:`(\\tfrac{1}{B}\\sum_i \\|g_i\\|^2 - \\|G\\|^2)/\\|G\\|^2`
with :math:`G = \\bar g` the mean gradient. Operates on the raw loss gradient
∇L. See ``docs/research/metrics.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from metrics.primitives import EPS, per_sample_grad_matrix


def _gns_core(G: torch.Tensor) -> dict[str, float]:
    """Simple gradient noise scale from a ``[M, P]`` per-sample gradient matrix."""
    Gbar = G.mean(0)
    gbar_sq = Gbar.dot(Gbar)
    tr_sigma = (G * G).sum(1).mean() - gbar_sq
    return {
        "noise_scale/noise": float(tr_sigma),
        "noise_scale/simple": float(tr_sigma / (gbar_sq + EPS)),
    }


class GnsSimpleMetric:
    name = "gns_simple"

    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        model.eval()
        G = per_sample_grad_matrix(model, X, y, loss_fn)
        return _gns_core(G)


METRIC = GnsSimpleMetric()
