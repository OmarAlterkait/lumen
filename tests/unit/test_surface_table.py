"""The guard of the surface table: the one kernel against the per-type formulas it replaced, bit for bit, per row.

Every row the build can produce (each of the twelve types and the structures derived from them, with random
parameters, all in one table as a detector packs it; and every row of the toy detectors) is evaluated through the
photon loop's path (``lumen.transport.surface_event.type_probs`` on the packed table) and by the frozen per-type
formulas (``surface_reference``), on random crossings, unpolarised and polarised. Both
run op by op (``jax.disable_jit``): a fused program's float bits depend on the compiler's fusion (XLA contracts a * b
+ c), the formulas' do not. The one exception is measured, not hidden: ``transmitting`` reads Geant4's cuts (R, T)
directly where the per-type code rebuilt R from its own absorbed share first (``test_transmitting_is_geant4s_cuts``).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import surface_reference as ref
from surface_reference import DETECT

import lumen
from lumen.build import lookup
from lumen.rules.entries import SensorEntry, SurfaceEntry
from lumen.surfaces import types
from lumen.surfaces.outcomes import FAR_SIDE, N_OUTCOMES, STRAIGHT
from lumen.transport import surface_event

N = 128
F32 = np.float32
FIELDS = types.TypeEval._fields
RANGES = {"R": (0, 1), "T": (0, 1), "eff": (0, 1), "f_spec": (0, 1), "p_spike": (0, 1), "p_lobe": (0, 1),
          "p_back": (0, 1), "sigma_alpha": (0, 0.6), "polish": (0, 1), "nr": (1.0, 3.0), "nk": (0.0, 2.0),
}  # fmt: skip


def _metals():
    return (types.GLISUR_METAL, types.UNIFIED_METAL, types.UNIFIED_METAL_NORMAL, types.SPECULAR_METAL,
            types.PHOTOCATHODE)  # fmt: skip


def structures():
    """Return every structure the rules derive (``lumen.rules.optics``, ``lumen.rules.boundary``, the library)."""
    t = types
    out = list(t.TYPES) + [
        t.PAINTED,
        t.REFLECTOR,
        t.SPECULAR_METAL,
        t.fixed(t.PAINTED_T, {"f_spec": 0.0}),
        t.fixed(t.PAINTED_T, {"f_spec": 1.0, "T": 0.0}),
        t.fixed(t.UNIFIED_METAL_NORMAL, {"p_back": 0.0}),
        t.fixed(t.COUNTING, {"eff": 1.0}),
    ]
    out += [t.no_detection(k) for k in list(out) if "eff" in k.params]
    out += [t.per_normal(k) for k in t.TYPES if k.detects]
    out += [t.fixed(t.per_normal(t.DIELECTRIC_RT), {"eff": 1.0})]
    out += [t.into_opaque(k) for k in list(out) if any(FAR_SIDE[o] for o in k.outcomes) and not k.refracts]
    return list(dict.fromkeys(out))  # equal structures are one


def transmitting():
    return [types.transmitting(k) for k in _metals()] + [types.into_opaque(types.transmitting(types.UNIFIED_METAL))]


def params(stype, g):
    """Return random float32 parameters of a row of ``stype``."""
    return {q: F32(g.uniform(*RANGES[q])) for q in stype.params}


def contexts(g, polarised):
    nrm = g.normal(size=(N, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    wi = g.normal(size=(N, 3))
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    wi = np.where(np.sum(wi * nrm, 1, keepdims=True) > 0, -wi, wi)
    wi[: N // 16] = -nrm[: N // 16]  # normal incidence
    ctx = dict(lam=g.uniform(150.0, 800.0, N), wi=wi, n=nrm, n1=g.uniform(1.0, 1.8, N), n2=g.uniform(1.0, 1.8, N))
    if polarised:
        a = np.cross(wi, g.normal(size=(N, 3)))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        th = g.uniform(0, 2 * np.pi, (N, 1))
        ctx["er"], ctx["ei"] = np.cos(th) * a, np.sin(th) * 0.3 * np.cross(wi, a)
    return {k: jnp.asarray(v, jnp.float32) for k, v in ctx.items()}


def table_of(entries):
    """Return (tables, packed columns) of a table whose rows are ``entries`` (name -> SurfaceEntry), as compile does."""
    lib = lookup.Library(surfaces={n: e for n, e in entries.items() if isinstance(e, SurfaceEntry)},
                         sensors={n: e for n, e in entries.items() if isinstance(e, SensorEntry)})  # fmt: skip
    rows = tuple(lookup.Row(e.surface_type, f"{'sensor' if isinstance(e, SensorEntry) else 'surface'}:{n}")
                 for n, e in entries.items())  # fmt: skip
    optics = SimpleNamespace(rows=rows, table=lookup._surface_table(lib, rows), media_names=())
    named = {"media": {}, "surfaces": {}, "sensors": {}}
    for n, e in entries.items():
        named["sensors" if isinstance(e, SensorEntry) else "surfaces"][n] = dict(e.params)
    pk = jax.tree_util.tree_map(jnp.asarray, lookup.pack(optics, named))["surfaces"]
    return optics, pk


def evaluate(optics, pk, i, stype, p, ctx):
    """Return (the table's TypeEval of row i, the per-type formulas'), per crossing as the photon loop evaluates it."""
    new = jax.vmap(lambda c: surface_event.type_probs(optics, pk, jnp.int32(i), c))(ctx)
    axes = {k: 0 if np.shape(v)[:1] == (N,) else None for k, v in p.items()}  # a law's value: per crossing
    old = jax.vmap(lambda c, q: ref.evaluate(stype, q, c), in_axes=(0, axes))(ctx, p)
    old = split_deposits(old)
    old = tuple(
        jnp.broadcast_to(jnp.asarray(y, jnp.asarray(x).dtype), jnp.shape(x)) for x, y in zip(new, old, strict=True)
    )
    return new, old


def split_deposits(old):
    """Return the per-type formulas' TypeEval: their six continuing shares and their DETECT share as the detected share
    (``TypeEval.detect``; the loss is the rest)."""
    pr = old[0]
    return (pr[..., :N_OUTCOMES], pr[..., DETECT], *old[1:])


def bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a = a.astype(np.float32) if a.dtype != np.bool_ else a
    b = b.astype(np.float32) if b.dtype != np.bool_ else b
    return a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def compare(stypes, seed, skip=()):
    """Return the (type, field, polarised) that differ between the table's kernel and the per-type formulas."""
    g = np.random.default_rng(seed)
    entries = {f"s{i}": (SensorEntry if k.detects else SurfaceEntry)(k, params(k, g))
               for i, k in enumerate(stypes)}  # fmt: skip
    optics, pk = table_of(entries)
    bad = []
    with jax.disable_jit():
        for i, e in enumerate(entries.values()):
            for pol in (False, True):
                ctx = contexts(g, pol)
                new, old = evaluate(
                    optics, pk, i, e.surface_type, {k: jnp.asarray(v) for k, v in e.params.items()}, ctx
                )
                for f, x, y in zip(FIELDS, new, old, strict=True):
                    if f not in skip and not bits_equal(x, y):
                        bad.append((e.surface_type.name, f, pol))
    return bad


@pytest.mark.parametrize("seed", [1, 2])
def test_every_structure_is_the_per_type_formula_bit_for_bit(seed):
    """Each structure, all in one table (the kernel's Fresnel switch over every kind), with random parameters."""
    assert compare(structures(), seed) == []


def test_transmitting_is_geant4s_cuts():
    """A metal with TRANSMITTANCE: the reflected shares, amplitudes and lobes are the per-type code's bit for bit; the
    straight share and the detect deposit are Geant4's cuts (R, T) read directly, which the per-type code reached
    through 1 - its own absorbed share (a rounding of R): equal to it within float32 rounding."""
    stypes = transmitting()
    assert compare(stypes, 3, skip=("probs", "detect")) == []
    g = np.random.default_rng(4)
    entries = {f"s{i}": SensorEntry(k, params(k, g)) for i, k in enumerate(stypes)}
    optics, pk = table_of(entries)
    with jax.disable_jit():
        for i, (_, e) in enumerate(entries.items()):
            ctx = contexts(g, True)
            new, old = evaluate(optics, pk, i, e.surface_type, {k: jnp.asarray(v) for k, v in e.params.items()}, ctx)
            eps = 4 * np.finfo(np.float32).eps
            (pn, dn), (po, do) = ((np.asarray(x, np.float64) for x in v[:2]) for v in (new, old))
            keep = [o for o in range(N_OUTCOMES) if o != STRAIGHT]
            assert np.array_equal(pn[:, keep], po[:, keep])
            np.testing.assert_allclose(pn, po, rtol=0, atol=eps)
            np.testing.assert_allclose(dn, do, rtol=0, atol=eps)
            assert np.all(pn.sum(-1) + dn <= 1.0 + eps)


@pytest.mark.parametrize("name", ["toy_sphere", "toy_cube", "toy_tank"])
def test_every_toy_detector_row_is_the_per_type_formula_bit_for_bit(name, tmp_path):
    """Every row of a toy detector, through its own packed table (its laws at the photon's wavelength)."""
    import builder
    import builder_toys

    det = lumen.load(builder.save(builder_toys.TOYS[name](), tmp_path / f"{name}.lumen"))
    o, lib = det.parts.optics, det.parts.library
    pk = jax.tree_util.tree_map(jnp.asarray, det.pack())["surfaces"]
    g = np.random.default_rng(5)
    with jax.disable_jit():
        for i, (stype, origin) in enumerate(o.rows):
            for pol in (False, True):
                ctx = contexts(g, pol)
                e = {} if origin is None else lookup._entry(lib, origin).params
                p = {k: (v[0](jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), v[1]), ctx["lam"])
                         if isinstance(v, tuple) else jnp.asarray(v, jnp.float32)) for k, v in e.items()}  # fmt: skip
                new, old = evaluate(o, pk, i, stype, p, ctx)
                for f, x, y in zip(FIELDS, new, old, strict=True):
                    assert bits_equal(x, y), (name, stype.name, f, pol)


def test_the_guard_discriminates():
    """A negative control: a one-ulp change of one column's value is seen."""
    k = types.no_detection(types.UNIFIED_METAL)
    g = np.random.default_rng(6)
    p = params(k, g)
    optics, pk = table_of({"s": SurfaceEntry(k, p)})
    pk = dict(pk, R=jnp.nextafter(pk["R"], 2.0))
    with jax.disable_jit():
        ctx = contexts(g, False)
        new, old = evaluate(optics, pk, 0, k, {q: jnp.asarray(v) for q, v in p.items()}, ctx)
    assert not bits_equal(new.probs, old[0])
