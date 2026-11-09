"""The physics tier's fixtures: the toys converted and built once per session, and one photon loop per toy shape.

``toy(name, *args)`` returns the toy's ``lumen.validate.photons.Setup`` (its detector converted from the GDML of
``toys.py`` into the session's directory, the navigator from the build cache), one Model per toy and step config, so
a test changing values never compiles the loop again (``toys.simulate`` keeps one ItemSpec per histogram shape).
"""

import hashlib

import pytest
import toys

import lumen
from lumen.transport import step as steps
from lumen.validate import photons


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Return toy(name, *args, step=StepConfig(), clearance=None, **kw) -> Setup, each built once per session."""
    root = tmp_path_factory.mktemp("toys")
    made = {}

    def get(name, *args, step=None, clearance=None, **kw):
        step = step or steps.StepConfig()
        gdml, recipe = getattr(toys, name)(*args, **kw)
        digest = hashlib.sha256(gdml.encode()).hexdigest()[:16]
        if digest not in made:
            src = root / f"{recipe.name}_{digest}.gdml"
            src.write_text(gdml)
            made[digest] = lumen.load(lumen.convert(src, recipe, root / f"{recipe.name}_{digest}.lumen"))
        key = (digest, step, id(clearance))
        if key not in made:
            made[key] = (photons.setup(made[digest], step=step, clearance=clearance), clearance)
        return made[key][0]

    return get
