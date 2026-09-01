"""Metric registry.

Each metric module exposes a module-level ``METRIC`` instance with a ``name``
and a ``compute(...)`` method (see :class:`metrics.base.Metric`). They are
collected into two groups because their call signatures differ:

* ``REGISTRY``: the gradient metrics,
  ``metric.compute(model, X, y, loss_fn) -> dict[str, float]``.
* ``BASELINE``: the TSE baseline, ``BASELINE.compute(losses) -> dict``, which
  takes a sequence of **per-epoch mean** training losses instead of a model
  (aggregate per-step losses first, see ``train.epoch_mean_losses``).
"""

from . import base, primitives  # noqa: F401
from .gns_simple import METRIC as gns_simple
from .gradient_confusion import METRIC as gradient_confusion
from .gradient_disparity import METRIC as gradient_disparity
from .gsnr import METRIC as gsnr
from .gwa import METRIC as gwa
from .m_coherence import METRIC as m_coherence
from .normalized_variance import METRIC as normalized_variance
from .stiffness import METRIC as stiffness
from .tse import METRIC as tse

# Uniform compute(model, X, y, loss_fn) interface, grouped by family.
REGISTRY = {
    m.name: m
    for m in (
        # variability family
        normalized_variance,
        gns_simple,
        gsnr,
        # alignment / coherence family
        m_coherence,
        stiffness,
        gradient_disparity,
        gradient_confusion,
        gwa,
    )
}

# Kept out of REGISTRY: its compute() takes a loss sequence, not a model.
BASELINE = tse

__all__ = ["REGISTRY", "BASELINE", "base", "primitives"]
