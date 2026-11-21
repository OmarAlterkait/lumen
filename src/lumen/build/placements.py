"""Geant4's navigation preconditions, tested on the navigator: overlapping siblings, protrusions, meshes.

Geant4 takes its placements as overlap-free (daughters inside their mothers, siblings disjoint); G4PVPlacement's
CheckOverlaps tests it with points on each volume's surface. ``check`` does the same with points just inside each
volume's surface (random lines through its own box: the entry point, stepped in by twice the placement check's step
in, ``lumen.rules.navigation.check_step_in``, the contact tolerance; a point counts when its six axis neighbours one
step in away are inside the volume too): a sibling holding all seven is an overlap (its region the owner's, the first
in the rule's order, ``lumen.geometry.ownership``), a mother holding none of them a protrusion, and a volume the tree
locates the point in that is neither the volume nor below it, nor an owner of it, an overlap across levels.

    c = check(nav)                     # dict(pairs, counts, npts, protrude, n_points, n_tried, cross, ...)
    m = check_meshes(nav)              # dict(pairs {(mesh, volume): points}, protrude {mesh: points}, n_points)
    for line in report(nav, c, m): print(line)
    lines, found = conversion(df, sensor_set)   # what lumen.convert records in the detector file's report

The checks report; they never change the navigator.

A check runs one program per call, the navigator's arrays its constants: a conversion builds a fresh navigator and
checks it once.
"""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from lumen import run as engine
from lumen.build import lowering
from lumen.geometry import navigate
from lumen.geometry.csg import pick
from lumen.rules import navigation as rules

F32, I32 = jnp.float32, jnp.int32
N_POINTS = 1000  # points per definition (kind x placement), spread over its instances
N_MIN = 16  # ... at least this many per instance
K_MAX = 4  # holders recorded per point
CHUNK = 1 << 20  # points per device call
MESH_POINTS = 16  # points per mesh chunk
SHOWN = 8  # the report shows at most this many families


def _backend(backend):
    """Return the resolved backend (``lumen.run.KERNEL``: a tallymap thread per point; else jax.vmap) of a name."""
    name = "auto" if backend is None else backend
    engine.check_device(name)
    return engine.resolve_backend(name)[0]


def runner(one, backend=None):
    """Return a function of arrays (i, id) running one(i, id) -> tuple over them.

    ``backend``: lumen's backend name (``lumen.run.resolve_backend``; None: "auto"): on "auto", "cuda" and "cpu" one
    thread of tallymap's kernel backend per point (on the GPU or the CPU; "cpu" always on the CPU), on "xla" jax.vmap.
    """
    if _backend(backend) == engine.KERNEL:

        def item(tl, x):
            return tl, {f"o{j}": v for j, v in enumerate(one(x["u"], x["id"]))}

        f = jax.jit(lambda xs: engine.vmap_runner(item, {"z": engine.sum_output(1)}, engine.KERNEL)(xs))

        def run(u, i):
            r = engine.on_device(backend, f, dict(u=u, id=i))  # "cpu": on the CPU, also on a GPU machine
            return tuple(r.per_item[f"o{j}"] for j in range(len(r.per_item)))

        return run

    return jax.jit(jax.vmap(one))


def _probe(nav):
    """Return the placement check's own arrays (``NavArrays.probe``: the volumes' boxes, the chunks' mothers)."""
    return {k: jnp.asarray(v) for k, v in nav.arrays.probe.items()}


def _point(nav, P, u, iid, seed):  # noqa: N803 -- P the probe arrays
    """Return (the line through volume u's box: origin, direction; its length) of point iid."""
    D = nav.dev()
    key = jax.random.fold_in(jax.random.PRNGKey(seed), iid)
    k1, k2 = jax.random.split(key)
    p = P["uc"][u] + (jax.random.uniform(k1, (3,), F32) * 2 - 1) * P["uh"][u]
    dl = jax.random.normal(k2, (3,), F32)
    dl = dl / jnp.linalg.norm(dl)
    Lr = F32(2.0) * jnp.linalg.norm(P["uh"][u]) + F32(1e-6)
    R_, t_ = D["R"][u], D["t"][u]
    return R_ @ (p - Lr * dl) + t_, R_ @ dl, Lr


def _sample(nav, P, u, iid, opts):  # noqa: N803 -- P the probe arrays
    """Return (ok, the point just inside volume u's surface) of point iid."""
    seed, tol_ = opts
    o, d, Lr = _point(nav, P, u, iid, seed)
    h1, te, in1, _, _, _ = nav.first_change_in(u, o, d, F32(0.0), F32(2.0) * Lr)
    h2, tx, _, _, _, _ = nav.first_change_in(u, o, d, te, F32(2.0) * Lr)
    xe = o + te * d
    res = rules.check_step_in(xe) if tol_ is None else tol_
    stp = jnp.minimum(2 * res, 0.5 * jnp.where(h2, tx - te, 4 * res))  # twice the step in: its neighbours one in
    xi = o + (te + stp) * d
    return h1 & in1 & nav.holds_point(u, xi), xi


STEPS = jnp.asarray(np.vstack([np.zeros(3), np.eye(3), -np.eye(3)]), F32)  # a point and its six axis neighbours


def _holds_all(nav, i, pts):
    """Return whether volume i holds every point of pts (7, 3)."""
    return jnp.all(jax.vmap(lambda x: nav.holds_point(i, x))(pts))


def _holders(nav, u, xi, kmax, pts):
    """Return the siblings of u holding xi deeper than the contact tolerance (xi and its six axis neighbours ``pts``):
    up to kmax indices, -1 padded."""  # noqa: D205, D209
    D = nav.dev()
    n = nav.n

    def visit(i, a):
        ic = jnp.minimum(i, n - 1)
        hit_ = (i != u) & (i < n) & (D["parent"][ic] == D["parent"][u]) & nav.holds_point(ic, xi)
        hit_ = hit_ & pick(hit_, lambda: _holds_all(nav, ic, pts), lambda: jnp.bool_(False))
        slot = jnp.argmax(a < 0)
        return jnp.where(hit_ & (a[slot] < 0), a.at[slot].set(i), a)

    return nav.scan(D["parent"][u], xi, visit, jnp.full((kmax,), -1, I32), lambda a: jnp.all(a >= 0))


def _across(nav, u, xi):
    """Return (where the tree locates xi, whether that is neither u nor below it)."""
    loc = nav.locate(xi)
    return loc, (loc != u) & ~nav.is_ancestor(u, loc)


def _one(nav, P, opts):  # noqa: N803 -- P the probe arrays
    """Return the per-point test: (ok, protrudes, holders, the across-levels holder)."""
    D = nav.dev()
    n = nav.n
    seed, kmax, tol = opts
    tol_ = None if tol is None else F32(tol)

    def one(u, iid):
        ok, xi = _sample(nav, P, u, iid, (seed, tol_))
        # a contact is no overlap: only a point deeper than the contact tolerance (the step in) inside u counts, its
        # six axis neighbours that far away inside too
        res = rules.check_step_in(xi) if tol_ is None else tol_
        pts = xi[None] + res * STEPS
        deep = pick(ok, lambda: _holds_all(nav, u, pts), lambda: jnp.bool_(False))
        pv = D["parent"][u]
        pc = jnp.minimum(pv, n - 1)
        out_m = (pv < n) & deep & ~jnp.any(jax.vmap(lambda x: nav.holds_point(pc, x))(pts))
        hold = jnp.where(deep, _holders(nav, u, xi, kmax, pts), -1)
        loc, away = _across(nav, u, xi)
        # the tree locates xi elsewhere: across levels -- unless an owner of u (or of an ancestor) takes it
        cross = ok & away
        return ok, ok & out_m, jnp.where(ok, hold, -1), jnp.where(cross, loc, -1)

    return one


def _pairs(a, b):
    if len(a):
        return np.unique(np.stack([a, b], 1), axis=0, return_counts=True)
    return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)


def check(nav, n_pts=N_POINTS, n_min=N_MIN, seed=0, kmax=K_MAX, tol=None, backend=None, chunk=CHUNK):  # noqa: PLR0913 -- the check's settings
    """Return Geant4's preconditions tested at points just inside each volume's surface.

    n_pts points per definition (kind x placement), spread over its instances, at least n_min per instance: a volume
    placed 25000 times is not sampled 25000x as densely as one placed once (a definition: its logical volume, its
    variants and pieces one). The pieces of one split union (``Definition.piece_of``) are one solid, and a solid whose
    holes the lowering dropped and a sibling filling one (``NavArrays.owned``'s forced pairs) are disjoint in the
    detector file: neither pair is reported. ``pairs`` are the real overlaps, the ones ``lumen.compare.ownership``
    rewrites. Returns dict(pairs (m, 2) (earlier, later), counts, npts (points per volume), protrude
    (volumes), n_points, n_tried, cross (volume, holder) and cross_counts, both without the points an owner takes;
    thin (m, 2): the sibling pairs that may share a point, not deeper than the contact tolerance, from the
    navigator's build (``NavArrays.thin``) -- touching for the navigator, so an overlap thinner than the tolerance is
    not one here).
    """
    n = nav.n
    opts = (int(seed), int(kmax), None if tol is None else float(np.float32(tol)))

    run = runner(_one(nav, _probe(nav), opts), backend)
    _, inv, cnt = np.unique(list(nav.origins[:n]), return_inverse=True, return_counts=True)
    npts = np.maximum(n_min, -(-n_pts // cnt[inv])).astype(np.int64)
    u_all = np.repeat(np.arange(n, dtype=np.int32), npts)
    ids = np.arange(len(u_all), dtype=np.int32)
    ok, prot, hold, cross = [], [], [], []
    for s0 in range(0, len(ids), chunk):
        a, b, c, e = run(jnp.asarray(u_all[s0 : s0 + chunk]), jnp.asarray(ids[s0 : s0 + chunk]))
        ok.append(np.asarray(a))
        prot.append(np.asarray(b))
        hold.append(np.asarray(c))
        cross.append(np.asarray(e))
    ok, prot, hold, cross = np.concatenate(ok), np.concatenate(prot), np.concatenate(hold), np.concatenate(cross)
    cs = cross >= 0
    cs[cs] = ~_taken(nav, u_all[cs], cross[cs])
    cpr, ccnt = _pairs(u_all[cs], cross[cs])
    a = np.repeat(u_all[:, None], kmax, 1)
    sel = hold >= 0
    pa, pb = a[sel], hold[sel]
    pc = np.asarray(nav.pieces, np.int64)
    one = (pc[pa] >= 0) & (pc[pa] == pc[pb])
    pr, cnt = _pairs(np.minimum(pa, pb)[~one], np.maximum(pa, pb)[~one])  # a split union's pieces are one solid
    # a lowered hole and its filler (forced): the detector file has the hole, so they do not overlap
    hole = {(min(a, b), max(a, b)) for a, b, f in nav.owned if f}
    real = np.array([(int(a), int(b)) not in hole for a, b in pr], bool).reshape(-1)
    pr, cnt = pr[real], cnt[real]
    return dict(pairs=pr, counts=cnt, npts=npts, protrude=np.unique(u_all[prot]), n_points=int(ok.sum()),
                n_tried=len(ok), cross=cpr, cross_counts=ccnt,
                thin=np.asarray(nav.thin, np.int64).reshape(-1, 2))  # fmt: skip


def _taken(nav, us, vs):
    """Return per (volume, holder) pair of ``us``, ``vs`` whether the holder is an owner the volume sees (``nav.kids``
    beyond its daughters), or below one -- a point of the volume the tree locates there is the owner's by the rule --
    or the volume is not proved inside its mother (a point of it beyond the mother is not its own). Per pair, from
    each volume's few candidates and ``size`` (the volumes below each, contiguous): no (volume, volume) table."""  # noqa: D205, D209
    par, size, n = np.asarray(nav.host["parent"]), np.asarray(nav.host["size"]), nav.n
    clipped = {int(c[0]) for c in nav.clipped if c[0] < n}
    owners: dict = {}
    out = np.zeros(len(us), bool)
    for i, (u, v) in enumerate(zip(us.tolist(), vs.tolist(), strict=True)):
        if u in clipped:
            out[i] = True
            continue
        if u not in owners:
            owners[u] = [k for k in nav.kids[u] if k < n and par[k] != u]
        out[i] = any(w <= v < w + int(size[w]) for w in owners[u])
    return out


def _mesh_one(nav, P, seed, kmax):  # noqa: N803 -- P the probe arrays
    D = nav.dev()
    n = nav.n

    def one(c, iid):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), iid)
        k1, k2 = jax.random.split(key)
        k = D["cstart"][c] + jnp.minimum((jax.random.uniform(k1, (), F32) * D["cn"][c]).astype(I32), D["cn"][c] - 1)
        u = jax.random.uniform(k2, (2,), F32)
        u = jnp.where(u[0] + u[1] > 1, 1 - u, u)
        p = D["V0"][k] + u[0] * (D["V1"][k] - D["V0"][k]) + u[1] * (D["V2"][k] - D["V0"][k])
        mv = P["cmoth"][c]
        out_m = (mv < n) & ~nav.holds_point(jnp.minimum(mv, n - 1), p)

        def visit(i, a):
            hit_ = (i < n) & nav.holds_point(jnp.minimum(i, n - 1), p)
            slot = jnp.argmax(a < 0)
            return jnp.where(hit_ & (a[slot] < 0), a.at[slot].set(i), a)

        hold = nav.scan(mv, p, visit, jnp.full((kmax,), -1, I32), lambda a: jnp.all(a >= 0))
        return out_m, hold, jnp.int32(0)

    return one


def check_meshes(nav, n_pts=MESH_POINTS, seed=0, kmax=K_MAX, backend=None):
    """Return the mesh check: n_pts points on each triangle chunk (a random triangle, a random barycentric point).

    A daughter of the mesh's mother holding one: the mesh passes through that volume (Geant4 does not see it from
    inside); the mother not holding one: the mesh protrudes. Returns dict(pairs {(mesh, volume): points}, protrude
    {mesh: points}, n_points).
    """
    if not nav.n_mesh:
        return dict(pairs={}, protrude={}, n_points=0)
    nch = len(nav.host["cmesh"])
    c_all = np.repeat(np.arange(nch, dtype=np.int32), n_pts)
    opts = (int(seed), int(kmax))

    run = runner(_mesh_one(nav, _probe(nav), *opts), backend)
    prot, hold, _ = run(jnp.asarray(c_all), jnp.arange(len(c_all), dtype=I32))
    prot, hold = np.asarray(prot), np.asarray(hold)
    mesh_of = np.asarray(nav.host["cmesh"])[c_all]
    pairs, pr = {}, {}
    for i, j in zip(*np.nonzero(hold >= 0), strict=False):
        key = (int(mesh_of[i]), int(hold[i, j]))
        pairs[key] = pairs.get(key, 0) + 1
    for i in np.flatnonzero(prot):
        pr[int(mesh_of[i])] = pr.get(int(mesh_of[i]), 0) + 1
    return dict(pairs=pairs, protrude=pr, n_points=len(c_all))


def _head(nav, i):
    """Return a navigator item's logical volume (``Navigator.origins``: per volume, the outside, then per mesh)."""
    return nav.origins[int(i)]


def _family(nav, a, b):
    return (_head(nav, a), _head(nav, b))


def _owned(nav, a, b):
    """Return (non-owner, owner) of an overlapping sibling pair: the owner the first in the rule's order."""
    pr = np.asarray(nav.host["prio"])
    return (a, b) if pr[b] < pr[a] else (b, a)


def report(nav, c, m=None):
    """Return the check's summary lines: the overlaps and their owners, protrusions, across levels, meshes."""
    pairs = np.asarray(c["pairs"]).reshape(-1, 2)
    fam = Counter(_head(nav, x) + " owned by " + _head(nav, y) for x, y in (_owned(nav, a, b) for a, b in pairs))
    lines = [f"overlaps ({c['n_points']} surface points): {len(pairs)} sibling pairs (points per pair "
             f"{np.sort(np.asarray(c['counts']))[::-1][:SHOWN].tolist()}) {dict(fam.most_common(SHOWN))}; "
             f"{len(c['protrude'])} protruding volumes "
             f"{dict(Counter(_head(nav, u) for u in c['protrude']).most_common(SHOWN))}"]  # fmt: skip
    cf = Counter()
    for (u_, v_), k_ in zip(c["cross"], c["cross_counts"], strict=False):
        cf[(_head(nav, u_), _head(nav, v_) if v_ < nav.n else "OUT")] += int(k_)
    lines.append(f"overlaps across levels (a point of a volume the tree locates outside it and its daughters): "
                 f"{len(c['cross'])} (volume, holder) pairs, {int(np.sum(c['cross_counts']))} points; "
                 f"{dict(cf.most_common(SHOWN))}")  # fmt: skip
    lines.append(_thin_line(nav, c))
    if m is not None and nav.n_mesh:
        lines.append(f"meshes ({m['n_points']} points): pass-throughs (mesh, volume): points "
                     f"{dict(sorted(((k, _head(nav, v)), x) for (k, v), x in m['pairs'].items())[:SHOWN])}; "
                     f"protruding meshes {m['protrude']}")  # fmt: skip
    return lines


def _thin_line(nav, c):
    """Return the report line of the sibling pairs closer than the contact tolerance, not proved apart."""
    th = Counter((_head(nav, a), _head(nav, b)) for a, b in np.asarray(c["thin"]).reshape(-1, 2))
    return (
        "siblings closer than the contact tolerance, not proved apart (touching here: an overlap thinner than "
        f"the tolerance is not resolved) (pairs): {_shown(th, lambda k: f'{k[0]} and {k[1]}')}"
    )


def _shown(cnt, fmt=str):
    top = sorted(sorted(cnt.items()), key=lambda kv: -kv[1])[:SHOWN]
    body = ", ".join(f"{fmt(k)}: {v}" for k, v in top)
    return f"{len(cnt)} families, {sum(cnt.values())} in all{'' if len(cnt) <= SHOWN else ', most'}: {{{body}}}"


def conversion(df, sensor_set, backend="cpu"):
    """Return (report lines, found) of the placement check on a detector file's detector volume, as simulated.

    ``lumen.convert`` records the lines in the detector file's report and warns (``PlacementWarning``) when ``found``:
    overlapping siblings (their shared region is then its owner's, ``lumen.geometry.ownership``), daughters reaching
    out of their mother (they hold only what lies inside it) -- Geant4 run on the same GDML differs there;
    ``lumen.compare.ownership.export`` writes the GDML that agrees --, or meshes passing through volumes. A contact
    (a shared face within the contact tolerance) is no overlap. Run on the CPU (``backend``) so the report is the same
    on every machine.
    """
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=sensor_set.is_sensor)
    nav = navigate.Navigator.from_scene(lowering.split_unions(loaded.scene), cache=True)  # its ownership: kept for load
    c = check(nav, backend=backend)
    m = check_meshes(nav, backend=backend)
    own = Counter(_family(nav, *_owned(nav, a, b)) for a, b in np.asarray(c["pairs"]).reshape(-1, 2))
    prot = Counter(_head(nav, u) for u in c["protrude"])
    mesh = Counter((f"mesh {k}", _head(nav, v)) for (k, v) in m["pairs"])
    head = f"placement check ({c['n_points']} points just inside every volume's surface; Geant4's CheckOverlaps)"
    thin = [_thin_line(nav, c)] if len(c["thin"]) else []
    if not (own or prot or mesh or m["protrude"]):
        return [f"{head}: no overlapping siblings, no protruding daughters", *thin], False
    lines = [f"{head}: found"]
    if own:
        lines.append("overlapping siblings, the shared region owned by one of them (pairs): "
                     + _shown(own, lambda k: f"{k[0]} owned by {k[1]}"))  # fmt: skip
    if prot:
        lines.append(f"daughters reaching out of their mother, held only inside it (placements): {_shown(prot)}")
    if mesh:
        lines.append(f"meshes passing through a volume, the shared region owned by one of them (mesh, volume): "
                     f"{_shown(mesh)}")  # fmt: skip
    if m["protrude"]:
        lines.append(f"meshes reaching out of their mother, held only inside it (meshes): {m['protrude']}")
    return lines + thin, True


class PlacementWarning(UserWarning):
    """The detector's placements overlap (``conversion``): recorded in its report."""
