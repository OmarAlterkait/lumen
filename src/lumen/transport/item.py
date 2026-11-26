"""The tallymap item's transport side: what one photon's loop starts from and the health each step adds.

    t = lumen.run.trace(model, ops, rows, key, fields=..., capacity=...)  # the photon loop: deposit records + health

``start_photon(model, ops, row)``: the Photon of an emission row (the row is the item's own mapped input, so a
re-emitted photon keeps its row); a photon runs while ``alive``. Transport gives each step's Deposit, which the loop
appends as a deposit record (``lumen.run.record``), and its fates (``fates``: the loop totals them per photon and
adds each photon's once, when it ends); everything built from the
deposits is the readout's (``lumen.readout.fill``, from the records).
"""

from __future__ import annotations

import jax.numpy as jnp

from lumen.transport import state
from lumen.transport.step import geometry

# the fates a step adds to its photon's totals (lumen.api.Health.FATES; "truncated" is run's), and the weight the
# stuck rule pushed (not a fate)
HEALTH = ("lost", "abandoned", "escaped", "distance_cut", "absorbed_bulk", "absorbed_surface", "rouletted",
          "reweighted", "pushed")  # fmt: skip


def start_photon(model, ops, row):
    """Return the Photon of an emission row, located (its volume and class from the navigator)."""
    nav = geometry(model, ops).nav
    vol = nav.locate(jnp.asarray(row.x, nav.F))
    return state.start(row, vol, nav.dev()["cls"][vol], gen="gen" in model.local, position=nav.F)


def fates(dep):
    """Return a Deposit's fate weights (``HEALTH``, in order) as one (len(HEALTH),) array."""
    return jnp.stack([getattr(dep, k) for k in HEALTH])


HEALTH_OUTPUTS = {k: (1, True) for k in HEALTH}
