"""The physics tier's fixtures: the toys converted and built once per session, and one photon loop per toy shape.

``toy(name, *args)`` returns the toy's ``lumen.compare.photons.Setup`` (its detector converted from the GDML of
``toys.py`` into the session's directory, the navigator from the build cache), one Model per toy and step config, so
a test changing values never compiles the loop again (``toys.simulate`` keeps one readout per histogram shape).

Without a GPU every test here is skipped except the CPU target's (``test_cpu_target.py``); every test here is marked
``gpu`` otherwise.
"""

import hashlib
from pathlib import Path

import jax
import pytest
import toys

import lumen
from lumen.transport import step as steps

HERE = Path(__file__).parent
CPU_FILES = {"test_cpu_target.py"}  # the photon loop's CPU target: runs without a GPU (GitHub's CI)


def pytest_collection_modifyitems(config, items):
    """Mark the physics tier ``gpu`` and skip it without a GPU, except the CPU target."""
    skip = None if jax.default_backend() == "gpu" else pytest.mark.skip(reason="the physics tier runs on a GPU")
    for item in items:
        if item.path.parent == HERE and item.path.name not in CPU_FILES:
            item.add_marker(pytest.mark.gpu)
            if skip is not None:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Return toy(name, *args, step=StepConfig(), **kw) -> Setup, each built once per session."""
    root = tmp_path_factory.mktemp("toys")
    made = {}

    def get(name, *args, step=None, **kw):
        step = step or steps.StepConfig()
        gdml, recipe = getattr(toys, name)(*args, **kw)
        digest = hashlib.sha256(gdml.encode()).hexdigest()[:16]
        if digest not in made:
            src = root / f"{recipe.name}_{digest}.gdml"
            src.write_text(gdml)
            made[digest] = lumen.load(lumen.convert(src, root / f"{recipe.name}_{digest}.lumen", recipe=recipe))
        key = (digest, step)
        if key not in made:
            made[key] = toys.setup(made[digest], step=step)
        return made[key]

    return get
