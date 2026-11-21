"""The surface types' formulas as lumen evaluated them per type before the one kernel (frozen: the guard's reference).

``tests/unit/test_surface_table.py`` checks ``lumen.surfaces.types.kernel`` against these bit for bit on every row the
build can produce. They are the per-type ``probs`` / ``amps`` / ``lobe_params`` of ``lumen.surfaces.types`` at lumen
commit 1c5a115, the composition of its deriving functions (``fixed`` merges its constants into the parameters,
``transmitting``, ``into_opaque``, ``per_normal``) and ``normal_absorptance``, kept here unchanged so the
reference cannot drift with the kernel (the GLG4 and thin-film photocathodes, deleted with their types, removed).
A type is named by its structure label (``SurfaceType.name``): its base type, then ``{constants}``, ``|T``,
``|into_opaque``, ``|per_normal`` (the layered ``coated`` ones deleted with their type).
"""

import re

import jax.numpy as jnp

from lumen._core.safe import literal
from lumen.surfaces import fresnel, outcomes, polar
from lumen.surfaces.outcomes import STRAIGHT
from lumen.surfaces.types import chained_cuts, cos_incidence, is_polarised

# the frozen formulas' own vector: the six continuing outcomes, then the detected share and the loss (the kernel gives
# the first six and the detected share; ``full`` and ``split`` convert)
DETECT, ABSORB = outcomes.N_OUTCOMES, outcomes.N_OUTCOMES + 1
N_FULL = outcomes.N_OUTCOMES + 2
NAMES = (*outcomes.NAMES, "detect", "absorb")
FAR_SIDE = (*outcomes.FAR_SIDE, False, False)


def outcome_vector(**shares):
    """Return the (8,) vector with the named shares (the six outcomes, "detect", "absorb") and zeros elsewhere."""
    v = [jnp.zeros((), jnp.float32) for _ in range(N_FULL)]
    for name, x in shares.items():
        v[NAMES.index(name)] = x
    return jnp.stack(jnp.broadcast_arrays(*v), -1)


def full(ev):
    """Return a kernel TypeEval's (8,) vector: its continuing shares, its detected share and the loss (the rest)."""
    d = ev.detect
    return jnp.concatenate([ev.probs, d[..., None], (1.0 - jnp.sum(ev.probs, -1) - d)[..., None]], -1)


DEFAULT_WIDTH = 0.1
MIRROR = (*polar.MIRROR, 1.0 + 0.0j, 1.0 + 0.0j)


def _fresnel(ctx):  # the amplitudes' reflectance (one formula for the dielectric, total reflection included)
    rs, rp, _, _ = fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], ctx["n2"])
    fs, fp = polar.fractions(ctx["er"], ctx["ei"], ctx["wi"], ctx["n"]) if is_polarised(ctx) else (0.5, 0.5)
    return jnp.clip(fresnel.reflectance(fs, fp, rs, rp), 0.0, 1.0)


def _dielectric(p, ctx):
    r = _fresnel(ctx)
    return outcome_vector(refract=1.0 - r, specular=r)


def _dielectric_amps(p, ctx):
    return fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], ctx["n2"])


def _dielectric_rt(p, ctx):
    r, t, a = chained_cuts(p["R"], p["T"])
    F = _fresnel(ctx)
    eff = p["eff"]
    return outcome_vector(refract=r * (1.0 - F), specular=r * F, straight=t, detect=a * eff, absorb=a * (1.0 - eff))


def _photocathode_amps(p, ctx):
    return fresnel.amplitudes(cos_incidence(ctx), ctx["n1"], p["nr"] + 1j * p["nk"])


def _photocathode(p, ctx):
    if is_polarised(ctx):
        rs, rp, _, _ = _photocathode_amps(p, ctx)
        fs_, fp_ = polar.fractions(ctx["er"], ctx["ei"], ctx["wi"], ctx["n"])
        R = jnp.clip(fresnel.reflectance(fs_, fp_, rs, rp), 0.0, 1.0)
    else:
        R = fresnel.unpolarised_absorbing_reflectance(cos_incidence(ctx), ctx["n1"], p["nr"], p["nk"])
    A = 1.0 - R
    return outcome_vector(specular=R, detect=A * p["eff"], absorb=A * (1.0 - p["eff"]))


def _counting(p, ctx):
    return outcome_vector(detect=p["eff"], absorb=1.0 - p["eff"])


def _glisur_metal(p, ctx):
    r, a = chained_cuts(p["R"])
    return outcome_vector(glossy=r, detect=a * p["eff"], absorb=a * (1.0 - p["eff"]))


def _unified_metal(p, ctx):
    r, a = chained_cuts(p["R"])
    ss, sl, bs, lam = chained_cuts(p["p_spike"], p["p_lobe"], p["p_back"])
    return outcome_vector(specular=r * ss, glossy=r * sl, backscatter=r * bs, diffuse=r * lam, detect=a * p["eff"],
                          absorb=a * (1.0 - p["eff"]))  # fmt: skip


def _unified_metal_normal(p, ctx):
    r, a = chained_cuts(p["R"])
    ss, sl, bs, lam = chained_cuts(p["p_spike"], p["p_lobe"], p["p_back"])
    return outcome_vector(specular=r * (ss + sl), backscatter=r * bs, diffuse=r * lam, detect=a * p["eff"],
                          absorb=a * (1.0 - p["eff"]))  # fmt: skip


def _painted_t(p, ctx):
    r, t, a = chained_cuts(p["R"], p["T"])
    return outcome_vector(specular=r * p["f_spec"], diffuse=r * (1.0 - p["f_spec"]), straight=t, detect=a * p["eff"],
                          absorb=a * (1.0 - p["eff"]))  # fmt: skip


# base name -> (probs, amps or None, lobe width or None)
BASES = {
    "pass": (lambda p, ctx: outcome_vector(straight=jnp.ones(())), None, None),
    "absorber": (lambda p, ctx: outcome_vector(absorb=jnp.ones(())), None, None),
    "dielectric": (_dielectric, _dielectric_amps, None),
    "dielectric_rt": (_dielectric_rt, _dielectric_amps, None),
    "photocathode": (_photocathode, _photocathode_amps, None),
    "counting": (_counting, None, None),
    "glisur_metal": (_glisur_metal, None, lambda p: 1.0 - p["polish"]),
    "unified_metal": (_unified_metal, None, lambda p: p["sigma_alpha"]),
    "unified_metal_normal": (_unified_metal_normal, None, None),
    "painted_t": (_painted_t, None, None),
}


def _transmitting(probs):
    def f(p, ctx):
        pr = probs(p, ctx)
        refl = jnp.clip(1.0 - pr[..., DETECT] - pr[..., ABSORB], 0.0, 1.0)
        _, t, a = chained_cuts(refl, p["T"])
        eff = p["eff"]
        return pr.at[..., STRAIGHT].set(t).at[..., DETECT].set(a * eff).at[..., ABSORB].set(a * (1.0 - eff))

    return f


def _into_opaque(probs):
    def f(p, ctx):
        pr = probs(p, ctx)
        far = jnp.asarray(FAR_SIDE)
        return jnp.where(far, 0.0, pr).at[..., ABSORB].add(jnp.sum(jnp.where(far, pr, 0.0), -1))

    return f


def _plain(stype):
    """Return (probs, amps, width) of a type that is no layered one, from its structure label and flags."""
    probs, amps, width = BASES[re.match(r"[a-z_0-9]+", stype.name).group(0)]
    if "|T" in stype.name:
        probs = _transmitting(probs)
    if stype.far_opaque:
        probs = _into_opaque(probs)
    return probs, amps, width


def _full(stype, p, prefix=""):
    """Return the parameters with the type's constants merged in (``fixed``: the parameters win), as old ``fixed``."""
    out = {}
    for k, v in stype.consts:
        if k.startswith(prefix):
            out[k[len(prefix) :]] = literal(v) if isinstance(v, tuple) else jnp.float32(v)
    return {**out, **{k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}}


def evaluate(stype, p, ctx):
    """Return (probs, sigma_alpha, amps8, dscale) of one row at one crossing (``TypeEval``'s order, deposits apart)
    as the per-type code gave them."""
    probs, amps, width = _plain(stype)
    q = _full(stype, p)
    pr = probs(q, ctx)
    dscale = 1.0
    if stype.per_normal:
        c = {k: v for k, v in ctx.items() if k not in ("er", "ei")}
        c["wi"] = -ctx["n"]
        a_n = probs({**q, "eff": jnp.float32(1.0)}, c)[..., DETECT]
        dscale = jnp.where(a_n > 0.0, 1.0 / jnp.where(a_n > 0.0, a_n, 1.0), 0.0).astype(jnp.float32)
    a = amps(q, ctx) if amps else MIRROR
    return pr, width(q) if width else DEFAULT_WIDTH, _amps8(a), dscale


def _amps8(a):
    a = [jnp.asarray(z, jnp.complex64) for z in a]
    return jnp.stack([f(z) for z in a for f in (jnp.real, jnp.imag)]).astype(jnp.float32)
