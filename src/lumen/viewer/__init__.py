"""A browser viewer of a GDML file as lumen reads it: the geometry, and what lumen does with each material and surface.

    python -m lumen.viewer tank.gdml --open
    view = lumen.viewer.build("tank.gdml", recipe="my_recipes.py:TANK")

Exact boolean solids need the ``view`` extra (``manifold3d``).
"""

from lumen.viewer.scene import View, build

__all__ = ["View", "build"]
