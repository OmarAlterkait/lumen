"""The stuck rule's push (``step._recover``): a photon pushed across a boundary is located there, so its next step
reads the medium of the side it is on (its class with its volume), not the one it left (review7 F2 finding 3)."""

import jax
import jax.numpy as jnp
import numpy as np
import toys

from lumen.rules import navigation
from lumen.transport import state
from lumen.transport import step as steps

F32 = jnp.float32


def test_a_pushed_photon_takes_the_class_of_where_it_lands(toy):
    s = toy("sphere", r=0.0, absorption=2.0)
    ops = steps.operands(s.model, s.det.pack(s.det.params()), None, steps.Controls())
    nav = steps.geometry(s.model, ops).nav
    cls = nav.dev()["cls"]
    x0 = jnp.asarray([0.0, 0.0, toys.A], F32)
    push = navigation.PUSH_DISTANCE_TOLERANCES * navigation.step_band(x0)
    x = x0 - 0.5 * push * jnp.asarray([0.0, 0.0, 1.0], F32)  # inside the water, half a push from the shell
    v = nav.locate(x)
    row = toys.pencil(np.asarray(x), (0.0, 0.0, 1.0), 400.0, 1)
    ph = state.start(jax.tree_util.tree_map(lambda a: a[0], row), v, cls[v])
    ph = ph._replace(nz=jnp.asarray(navigation.ZERO_STEPS_BEFORE_PUSH, jnp.int32))
    try:
        ph2, _ = steps._recover(s.model, ops, ph)
    except TypeError:  # the signature before the fix
        ph2, _ = steps._recover(ph)
    landed = nav.locate(ph2.x)
    assert int(cls[landed]) != int(cls[v])  # the push crossed into the shell
    assert int(ph2.cls) == int(cls[landed]), (int(ph2.cls), int(cls[landed]), int(cls[v]))
