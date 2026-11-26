"""The step guard's count on the hand-written YAML toy made lossless.

Every liner and cathode REFLECTIVITY 1 and the water's absorption length x 1e6: nothing absorbs, and there is no time
cut, so a lossless run is bounded only by the step guard (``lumen.run.MAX_VERTICES``); the weight it stops alive is
``health.truncated``. The reflectivities are defensive rows (keyed, at their edge): about 1 % of the draws at each
reflection are covered outcomes of live probability 0, whose photons go on with weight 0 while the others carry
1 / (1 - e), so the stopped weight is the emitted weight in expectation, not photon by photon.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import lumen
from lumen import run as runner
from lumen.format.calibrations import Table

TOY = Path(__file__).resolve().parents[2] / "examples" / "data" / "yaml_toy"
N, GUARD = 4096, 64


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    return lumen.load(lumen.convert(TOY, tmp_path_factory.mktemp("lossless") / "toy.lumen"))


def test_a_lossless_runs_stopped_weight_is_reported(det, monkeypatch):
    for f in (runner.run, runner.trace):
        monkeypatch.setitem(f.__kwdefaults__, "max_vertices", GUARD)
    p = det.params()
    for k in sorted(p._values):
        if k.endswith("/REFLECTIVITY"):
            v = p[k]
            p = p.set(k, Table(jnp.asarray(v.energy), jnp.ones_like(jnp.asarray(v.value))))
    p = p.scale("materials/Water/ABSLENGTH", 1e6)
    for k in [k for k in p._values if k.startswith("sensors/") and k.endswith("/qe")]:
        p = p.scale(k, 0.0)  # a cathode of R = 1 absorbs nothing: its QE is 0 (the run refuses a QE above A_n)
    src = lumen.sources.Point((0.05, 0.02, 0.1), n_photons=float(N), lam=420.0)
    light = lumen.simulate(det, src, p, samples=N, seed=3)
    h = {k: float(np.sum(np.asarray(v))) for k, v in light.health._asdict().items()}
    assert float(np.sum(np.asarray(light.arrivals))) == 0.0  # R = 1 everywhere: nothing absorbed, nothing detected
    assert h["lost"] == h["escaped"] == h["abandoned"] == 0.0, h
    # per photon the stopped weight is 0 or (1 / (1 - e))^k (k reflections, k <= GUARD): sd below 1, so the sum's
    # relative spread is below 1 / sqrt(N) = 1.6 %
    assert h["truncated"] == pytest.approx(N, rel=0.08), h
