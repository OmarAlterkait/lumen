"""Tools for derivatives and fits: the gradient check, named keys as a vector, the Fisher information, a fit, scans.

``check_gradient(f, theta0)``: any derivative (an extension's included) against finite differences on the same random
numbers (``lumen.tools.check``), judged by the pass rules of ``lumen.tools.stats``. ``Space``: named parameter keys as
one vector (``lumen.tools.space``); ``jacobian``, ``fisher``, ``fit``: Newton steps on a Poisson likelihood's score
from two independent runs, ``descend``: first-order steps on it, ``coverage``: toy refits, ``toy_counts``: one toy's
data from other seeds (``lumen.tools.fit``); ``poisson_nll`` (``lumen.tools.losses``); ``scan``, ``profile``
(``lumen.tools.scan``). ``lumen.tools.plot`` draws them (matplotlib).
A model is plain JAX from (a ``Params``, a seed) to expected counts, ``lumen.simulate`` then a readout.
"""

from lumen.tools.check import GradientReport, check_gradient
from lumen.tools.fit import CoverageResult, FitResult, coverage, descend, fisher, fit, jacobian, toy_counts
from lumen.tools.losses import poisson_nll
from lumen.tools.scan import ProfileResult, ScanResult, profile, scan
from lumen.tools.space import Space

__all__ = ["CoverageResult", "FitResult", "GradientReport", "ProfileResult", "ScanResult", "Space", "check_gradient",
           "coverage", "descend", "fisher", "fit", "jacobian", "poisson_nll", "profile",
           "scan", "toy_counts"]  # fmt: skip
