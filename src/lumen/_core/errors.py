"""Errors shared by layers that must not import each other (the sources and the parameter packing)."""


class GradientNotAvailable(ValueError):  # noqa: N818 -- the documented error catalogue's name
    """A derivative the method cannot give (docs/explanation/known-limits.md); public as ``lumen.sources``'s."""


class ModelError(ValueError):
    """A registered optics model whose output breaks the rules of model code (``lumen.models``)."""


class LibraryError(ValueError):
    """A recorded optics decision that cannot be turned into the build's optics library (``lumen.build.library``)."""


def refusing(message: str):
    """Return the identity on a value's leaves whose derivative raises ``GradientNotAvailable(message)``.

    The rule runs only for a tangent that is not a symbolic zero (``jax.custom_jvp(symbolic_zeros=True)``), so a value
    traced without a derivative (``jax.jit``, ``jax.vmap``) passes, and the refusal comes when a derivative is traced,
    under ``jax.jit`` too.
    """
    import jax
    import jax.numpy as jnp

    def rule(primals, tangents):
        if not isinstance(tangents[0], jax.custom_derivatives.SymbolicZero):
            raise GradientNotAvailable(message)
        return primals[0], jnp.zeros_like(primals[0])

    f = jax.custom_jvp(lambda a: a)
    f.defjvp(rule, symbolic_zeros=True)
    return lambda v: jax.tree_util.tree_map(f, v)
