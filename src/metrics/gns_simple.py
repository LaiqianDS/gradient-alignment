"""Gradient noise scale, simple per-sample estimator (McCandlish et al., 2018).

Implements the simplified form under :math:`H \\propto I`,
:math:`\\mathcal{B}_{\\text{simple}} = \\mathrm{tr}(\\Sigma)/\\|G\\|^2`, estimated
per-sample as :math:`\\tfrac{1}{M}\\sum_i \\|g_i - \\bar g\\|^2 / \\|\\bar g\\|^2`
(algebraically :math:`(\\tfrac{1}{M}\\sum_i \\|g_i\\|^2 - \\|\\bar g\\|^2)/\\|\\bar g\\|^2`,
but non-negative by construction in floating point). The numerator is the
plug-in estimator of :math:`\\mathrm{tr}(\\Sigma)` — biased low by
:math:`(M-1)/M`, deliberately uncorrected (≈0.4% at M=256; the paper itself
concedes the ratio is not unbiased, App. A.1 fn. 12). It is logged as
``noise_scale/tr_sigma``; the key ``noise_scale/noise`` is reserved in the
docs for the exact Hessian-weighted :math:`\\mathcal{B}_{\\text{noise}}`,
which was dropped for cost. Operates on the raw loss gradient ∇L.
See ``docs/research/metrics.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .primitives import EPS, per_sample_grad_matrix


def _gns_core(G: torch.Tensor) -> dict[str, float]:
    """Simple gradient noise scale from a ``[M, P]`` per-sample gradient matrix."""
    Gbar = G.mean(0)
    gbar_sq = Gbar.dot(Gbar)
    # mean_i ‖g_i - ḡ‖²: same algebra as mean_i ‖g_i‖² - ‖ḡ‖², but immune to
    # the fp32 cancellation that can turn the difference-of-squares negative.
    tr_sigma = (G - Gbar).pow(2).sum(1).mean()
    return {
        "noise_scale/tr_sigma": float(tr_sigma),
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
