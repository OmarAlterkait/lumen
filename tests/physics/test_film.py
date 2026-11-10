"""A one-sided WLS film through the step (tier b, D51): the film toy with its holder turned 90 degrees about z.

VUV light (128 nm, absorbed within 1e-11 mm by the film) from 5 cm in front of the film face is re-emitted into the
visible band; from 5 cm behind the foil's back face no light can reach the film (the tank's walls absorb), so no
visible light is detected at all, and the bare back face reflects VUV light (paint, R = 0.5) the film face absorbs.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from toys import simulate

import lumen
from lumen.validate import photons

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))  # the unit tier's film toy
from film_toy import RECIPE, film_gdml

N = 1 << 16
CENTRE_MM = (0, 0, 500)  # off the origin: the film (20 nm) is thinner than the band there (0.5 um), so lowered
VUV, SPLIT = 128.0, 300.0  # nm


@pytest.fixture(scope="module")
def film(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("film")
    (tmp / "film.gdml").write_text(film_gdml(((CENTRE_MM, (0, 0, 90)),)))
    det = lumen.load(lumen.convert(tmp / "film.gdml", RECIPE, tmp / "film.lumen"))
    return photons.setup(det, (RECIPE.wavelength_range[0], SPLIT, RECIPE.wavelength_range[1]))


def charge(s, side):
    """Return (VUV, visible) detected per photon from 5 cm off the film's face (side +1) or the back face (-1)."""
    (axis,) = s.det.tree.film_axes.values()
    assert np.allclose(axis, [0.0, -1.0, 0.0])  # the local +x face, turned 90 degrees about z
    x = jnp.tile(jnp.asarray(np.asarray(CENTRE_MM) * 1e-3 + 0.05 * side * np.asarray(axis), jnp.float32), (N, 1))
    rows = photons.isotropic(x, jnp.full(N, VUV, jnp.float32), jax.random.PRNGKey(3))
    q = np.asarray(simulate(s, photons.packed(s.det), rows)["Q"]).reshape(-1, 2).sum(0) / N
    return float(q[0]), float(q[1])


def test_only_the_film_face_re_emits_and_the_back_face_reflects(film):
    vuv_front, vis_front = charge(film, 1.0)
    vuv_back, vis_back = charge(film, -1.0)
    assert vis_back == 0.0 and vis_front > 0.05
    assert vuv_back > vuv_front > 0.0
