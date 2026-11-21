"""The photon record: what a source emits and the photon loop starts from, one definition for both.

A source's ``emit`` returns a ``Photons`` batch (``lumen.sources``) and the loop builds each photon's state from its
row (``lumen.transport.state.start``). A source function (``@lumen.source``) returns one too.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# A deposit's optional fields (``lumen.transport.step.Deposit``), each carried only when a run reads it: name -> (shape,
# dtype). In the frame of the sensor's placement ``pos`` the crossing point (m), ``dir`` the photon's direction,
# ``normal`` the surface's normal; ``gen`` the photon's re-emission count; ``inv_an`` 1 / A_n of the crossed surface;
# ``label`` the photon's emission label; ``share`` its reached share (truth rows).
DEPOSIT_FIELDS = {"pos": ((3,), jnp.float32), "dir": ((3,), jnp.float32), "normal": ((3,), jnp.float32),
                  "gen": ((), jnp.int32), "inv_an": ((), jnp.float32), "label": ((), jnp.int32),
                  "share": ((), jnp.float32)}  # fmt: skip
VECTOR_FIELDS = tuple(k for k, (shape, _) in DEPOSIT_FIELDS.items() if shape == (3,))  # the geometric ones


class Photons(NamedTuple):
    """n photons (a batch, one row each); ``pol`` and ``label`` are optional.

    ``x`` (n, 3) m, ``d`` (n, 3) unit, ``t`` (n,) ns, ``lam`` (n,) nm, ``w`` (n,) the photons each stands for (their sum
    in expectation the photons the source stands for), ``pol`` (n, 2, 3) the Jones vector's real and imaginary parts
    (None or zeros: unpolarised; the loop's rows always carry it), ``label`` (n,) int32 each photon's label (None: no
    labels; the deposit's ``label``: the event axis's event, truth rows' label; the row is the photon's for its whole
    life, so a re-emitted photon keeps it).
    """

    x: jax.Array
    d: jax.Array
    t: jax.Array
    lam: jax.Array
    w: jax.Array
    pol: jax.Array | None = None
    label: jax.Array | None = None
