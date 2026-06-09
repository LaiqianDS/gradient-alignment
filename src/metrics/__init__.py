"""Gradient-alignment metric registry.

Each metric lives in its own module exposing a module-level ``METRIC`` instance
with a ``name`` and a ``compute(...)`` method (see :class:`metrics.base.Metric`).
They are collected here into two groups, because their call signatures differ:

* ``REGISTRY`` — the eight gradient metrics. Each takes a *frozen* model and a
  data probe and returns scalars:
  ``metric.compute(model, X, y, loss_fn) -> dict[str, float]``.
* ``BASELINE`` — the mandatory TSE baseline predictor. It takes a sequence of
  training losses instead of a model: ``BASELINE.compute(losses) -> dict``.
  Every gradient metric must out-predict TSE-EMA to justify its cost, so it is
  kept separate on purpose (see ``docs/research/metrics.md``).

Example
-------
>>> from metrics import REGISTRY, BASELINE
>>> row = {}
>>> for name, metric in REGISTRY.items():        # one probe -> many scalars
...     row.update(metric.compute(model, X, y, loss_fn))
>>> row.update(BASELINE.compute(loss_history))   # baseline, different input
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

# Eight gradient metrics — uniform compute(model, X, y, loss_fn) interface.
# Grouped by family: stochastic variability, then directional alignment.
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

# Mandatory baseline predictor — compute(losses) interface, kept separate.
BASELINE = tse

__all__ = ["REGISTRY", "BASELINE", "base", "primitives"]
