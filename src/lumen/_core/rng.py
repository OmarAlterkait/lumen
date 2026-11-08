"""Counter-based random streams with declared layouts.

Every random number a photon uses is a pure function of (base key, photon id, vertex index, name): replaying a
vertex reproduces its draws exactly, and adding a draw to one piece of the step never shifts another piece's draws,
as long as each piece owns its named block in the ``Layout``.

Uniforms are in the OPEN interval (0, 1): rand = (k + 1/2) 2^-23 with k a 23-bit integer, exactly representable in
float32, so -log(rand), log(1 - rand), 1 / rand and inverse CDFs are finite.
"""

import jax
import jax.numpy as jnp

UNIFORM_BITS = 23  # mantissa bits of a float32: every uniform is exact in float32


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
    """Return n float32 uniforms in (0, 1), never 0 or 1."""
    bits = jax.random.bits(key, (n,), jnp.uint32)
    k = (bits >> (32 - UNIFORM_BITS)).astype(jnp.float32)
    return (k + 0.5) * jnp.float32(2.0**-UNIFORM_BITS)


def draw(key, layout):
    """Return the named uniform blocks of one vertex: {name: (size,) array}."""
    return layout.split(uniforms(key, layout.n))
