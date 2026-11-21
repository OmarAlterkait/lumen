"""Counter-based random streams with declared layouts.

Every random number a photon uses is a pure function of (base key, photon id, vertex index, name): replaying a
vertex reproduces its draws exactly, and adding a draw to one piece of the step never shifts another piece's draws,
as long as each piece owns its named block in the ``Layout``; a draw outside the vertices folds its own key with one
of the stream ids below.

Uniforms are in the OPEN interval (0, 1): rand = (k + 1/2) 2^-23 with k a 23-bit integer, exactly representable in
float32, so -log(rand), log(1 - rand), 1 / rand and inverse CDFs are finite.
"""

import jax
import jax.numpy as jnp
import numpy as np

UNIFORM_BITS = int(np.finfo(np.float32).nmant)  # mantissa bits of a float32: every uniform is exact in float32

MAX_PHOTONS = 2**31 - 1  # photon ids are int32 (the sources' first_id + arange): a run draws at most this many

# Every stream drawn outside a vertex's Layout: the id folded into a key to draw it, one table, no two alike. A photon's
# id is below MAX_PHOTONS and a vertex index below MAX_VERTICES (2^22): the photon-key folds are above both.
SOURCE_STREAM = np.uint32(0xFFFFFFFF)  # the run key's fold of the source's draws: never a photon's id
ANALOG_STREAM = 0x7A4A1061  # the photon key's fold of its analog uniform (lumen.run)
DETECT_STREAM = 0x7A4A1062  # the photon key's fold of its detection uniform (lumen.readout.truth)
JUMP_STREAM = 0x7A11  # the vertex key's fold that draws the diffusion jump's own uniforms (lumen.transport.fastpath)
POLARISATION_STREAM = 0x9017  # the vertex key's fold that draws an unpolarised photon's linear state
STREAMS = (int(SOURCE_STREAM), ANALOG_STREAM, DETECT_STREAM, JUMP_STREAM, POLARISATION_STREAM)
assert len(set(STREAMS)) == len(STREAMS)
# a facet's own key's fold of the GLISUR shell choice (lumen.surfaces.facets): that key's other folds are the redraws'
# uint32 counters, which reach it only after 2^32 draws; the same number as SOURCE_STREAM, which folds another key
SHELL_STREAM = 0xFFFFFFFF
# the same key's fold of a dielectric facet's Fresnel choice (reflected or refracted): another number the counters
# reach only after 2^32 - 1 draws
FRESNEL_STREAM = SHELL_STREAM - 1

# The step's uniforms per block that a plug-in or a record must fit: a lobe draws at most LOBE_UNIFORMS (a bulk lobe's
# check refuses more, lumen._core.checks), a re-emission REEMIT_UNIFORMS (its wavelength, then one per time-profile
# component: lumen.media.bulk.Channel refuses more)
LOBE_UNIFORMS = 3
REEMIT_UNIFORMS = 2


class Layout:
    """A static map from block name to its number of uniforms; hashable, so usable as a static argument."""

    def __init__(self, **sizes):
        self.sizes = tuple(sizes.items())
        offset, self.slices = 0, {}
        for name, size in self.sizes:
            self.slices[name] = (offset, int(size))
            offset += int(size)
        self.n = offset

    def split(self, rand):
        """Return {name: the block's uniforms} from the uniforms of one vertex (..., n)."""
        return {name: rand[..., o : o + s] for name, (o, s) in self.slices.items()}

    def __hash__(self):
        """Hash by the block names and sizes."""
        return hash(self.sizes)

    def __eq__(self, other):
        """Compare by the block names and sizes."""
        return isinstance(other, Layout) and self.sizes == other.sizes

    def __repr__(self):
        """Show the blocks and their sizes."""
        return f"Layout({', '.join(f'{k}={s}' for k, s in self.sizes)})"


def photon_key(base, photon_id):
    """Return the key of one photon from the base key and the photon id."""
    return jax.random.fold_in(base, photon_id)


def vertex_key(pkey, vertex):
    """Return the key of one vertex of a photon from the photon's key."""
    return jax.random.fold_in(pkey, vertex)


def uniforms(key, n):
    """Return n float32 uniforms in (0, 1), never 0 or 1.

    Drawn in threefry's partitionable mode, whatever the global setting: there uniform i is a function of the key and i
    alone, so a layout that grows leaves every earlier block's draws as they were (the module docstring). Without it
    the counters are split by n and every draw changes with n (measured: JAX 0.6.2, ``jax_threefry_partitionable``
    False).
    """
    with jax.threefry_partitionable(True):
        bits = jax.random.bits(key, (n,), jnp.uint32)
    k = (bits >> (32 - UNIFORM_BITS)).astype(jnp.float32)
    return (k + 0.5) * jnp.float32(2.0**-UNIFORM_BITS)


def draw(key, layout):
    """Return the named uniform blocks of one vertex: {name: (size,) array}."""
    return layout.split(uniforms(key, layout.n))
