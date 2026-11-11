"""Validation: the float64 references and the checks lumen is held to.

``check_gradient(f, theta0)``: any derivative (an extension's included) against finite differences on the same random
numbers (``lumen.validate.check``).
"""

from lumen.validate.check import GradientReport, check_gradient

__all__ = ["GradientReport", "check_gradient"]
