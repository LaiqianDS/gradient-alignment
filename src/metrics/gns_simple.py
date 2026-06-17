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

from .primitives import EPS, stream_grad_moments


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


def _gns_from_moments(S: torch.Tensor, Q: torch.Tensor, M: int) -> dict[str, float]:
    """``_gns_core`` from the streamed moments ``S = Σg_i``, ``Q = Σg_i²``.

    ``‖ḡ‖² = ‖S‖²/M²`` and ``tr Σ = (Σ‖g_i‖² − M‖ḡ‖²)/M = Q.sum()/M − ‖ḡ‖²``.
    The difference-of-squares form is cancellation-prone in fp32; ``S``/``Q`` are
    float64 (see :func:`stream_grad_moments`), matching the full-matrix core to
    ~1e-7 — within the wiring test's 1e-6 tolerance.
    """
    gbar_sq = S.dot(S) / (M * M)
    tr_sigma = Q.sum() / M - gbar_sq
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
        S, Q, M = stream_grad_moments(model, X, y, loss_fn)
        return _gns_from_moments(S, Q, M)

    def reduce(self, sweep) -> dict[str, float]:
        """Same as :meth:`compute`, off the shared sweep (see ``metrics_runner``)."""
        return _gns_from_moments(sweep.S, sweep.Q, sweep.M)


METRIC = GnsSimpleMetric()
