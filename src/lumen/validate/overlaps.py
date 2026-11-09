"""Geant4's navigation preconditions, tested on the navigator: overlapping siblings, protrusions, meshes.

Geant4 takes its placements as overlap-free (daughters inside their mothers, siblings disjoint); G4PVPlacement's
CheckOverlaps tests it with points on each volume's surface. ``check`` does the same with points just inside each
volume's surface (random lines through its own box: the entry point, stepped in by the overlap check's step in,
``lumen.rules.navigation.check_step_in``): a sibling holding one is an overlap, a mother not holding one a protrusion,
and a volume the tree locates the point in that is neither the volume nor below it an overlap across levels.

    c = check(nav)                     # dict(pairs, counts, npts, protrude, n_points, n_tried, cross, cross_counts)
    m = check_meshes(nav)              # dict(pairs {(mesh, volume): points}, protrude {mesh: points}, n_points)
    for line in report(nav, c, m): print(line)

Notes
-----
Ported from the baseline's lumen.geometry.navigator (check, check_meshes, the runner) and lumen.geometry.tracked
(the report). The checks report; they never change the navigator (the baseline's links from them are not ported).

A check's program is compiled once and reused: with jax.vmap, per navigator layout (``layout``) and settings, the
navigator's arrays entering as arguments, so a repeated check, or a check of another navigator of the same layout,
traces nothing (a vmap of the navigator's nested loops takes seconds to trace); with tallymap, per navigator.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from lumen.rules import navigation as rules

F32, I32 = jnp.float32, jnp.int32
N_POINTS = 1000  # points per definition (kind x placement), spread over its instances
N_MIN = 16  # ... at least this many per instance
K_MAX = 4  # holders recorded per point
CHUNK = 1 << 20  # points per device call
MESH_POINTS = 16  # points per mesh chunk
TOUCH = 8  # a family (definition pair) holding at most this many of every 2000 points it contributed touches
TOUCH_PER = 2000
SHOWN = 8  # the report shows at most this many families
MAX_UP = 64  # the ancestor walk's depth limit
PIECE = re.compile(r"\|pc\d+(\||$)")  # a split union's piece definition
PIECE_PATH = re.compile(r"^(.*)/pc\d+$")
PROGRAMS = 8  # compiled checks kept, the least recently used dropped (each holds one navigator)
_PROGRAMS: OrderedDict = OrderedDict()  # (test, layout or navigator id, settings) -> (navigator, program)


def _backend(backend):
    if backend is not None:
        return backend
    try:
        import tallymap  # noqa: F401 -- only whether it imports

        return "tallymap" if jax.default_backend() == "gpu" else "vmap"
    except ImportError:
        return "vmap"


def runner(one, backend=None):
    """Return a function of arrays (i, id) running one(i, id) -> tuple over them.

    ``backend``: "tallymap" (one thread each; the default on a GPU where tallymap imports) or "vmap" (jax.vmap).
    """
    backend = _backend(backend)
    if backend == "tallymap":
        import tallymap as tm

        def item(tl, x):
            return tl, {f"o{j}": v for j, v in enumerate(one(x["u"], x["id"]))}

        f = jax.jit(lambda xs: tm.vmap(item, outputs={"z": tm.Sum((1,), exact=False)}, in_axes=0, backend="cuda")(xs))

        def run(u, i):
            r = f(dict(u=u, id=i))
            return tuple(r.per_item[f"o{j}"] for j in range(len(r.per_item)))

        return run

    return jax.jit(jax.vmap(one))


def layout(nav):
    """Return what a check's traced program depends on besides the values of the navigator's arrays.

    The Python values the navigator's device methods read (its sizes, the surface codes' layout, the CSG kernels
    present) and the shapes and dtypes of its arrays: navigators of one layout run one compiled check.
    """
    a, c = nav.arrays, nav.arrays.csg
    leaves, tree = jax.tree.flatten(nav.operands())
    return (a.n, a.n_mesh, a.chunk, int(a.ch0), tuple(a.srf), a.bg, c.n_kinds, c.maxnl, c.empty_leaf, c.has_torus,
            tuple(c.n_leaves), tree, tuple((x.shape, str(x.dtype)) for x in leaves))  # fmt: skip


def _program(test, nav, extra, settings, backend):
    """Return run(i, id) of a per-point test over arrays: ``test = (name, make)``, make(nav, extra) -> one(i, id).

    vmap: one program per (name, layout, settings), the navigator's arrays and ``extra`` (the per-navigator arrays
    the test reads) its arguments; tallymap: one per (name, navigator, settings), its arrays closed over.
    """
    name, make = test
    tallymap = _backend(backend) == "tallymap"
    key = (name, ("tallymap", id(nav)) if tallymap else layout(nav), settings)
    if key not in _PROGRAMS:
        if tallymap:
            prog = runner(make(nav, extra), "tallymap")
        else:

            def one(ops, i, iid, tmpl=nav):
                return make(tmpl.bind(ops["nav"]), ops["extra"])(i, iid)

            prog = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
        _PROGRAMS[key] = (nav, prog)  # the navigator held: an id in a key stays its own
        if len(_PROGRAMS) > PROGRAMS:
            _PROGRAMS.popitem(last=False)
    _PROGRAMS.move_to_end(key)
    prog = _PROGRAMS[key][1]
    if tallymap:
        return prog
    ops = dict(nav=nav.operands(), extra=extra)
    return lambda i, iid: prog(ops, i, iid)


def _piece_groups(nav):
    """Return per volume (and the outside) its split union's instance id (-1 none): the pieces of one placement."""
    pkey = [PIECE_PATH.match(nm_) if PIECE.search(df_) else None for nm_, df_ in zip(nav.names, nav.defs, strict=False)]
    keys = {}
    ids = [-1 if m_ is None else keys.setdefault(m_.group(1), len(keys)) for m_ in pkey]
    return jnp.asarray(np.array(ids + [-1], np.int32))


def _point(nav, u, iid, seed):
    """Return (the line through volume u's box: origin, direction; its length) of point iid."""
    D = nav.dev()
    key = jax.random.fold_in(jax.random.PRNGKey(seed), iid)
    k1, k2 = jax.random.split(key)
    p = D["uc"][u] + (jax.random.uniform(k1, (3,), F32) * 2 - 1) * D["uh"][u]
    dl = jax.random.normal(k2, (3,), F32)
    dl = dl / jnp.linalg.norm(dl)
    Lr = F32(2.0) * jnp.linalg.norm(D["uh"][u]) + F32(1e-6)
    R_, t_ = D["R"][u], D["t"][u]
    return R_ @ (p - Lr * dl) + t_, R_ @ dl, Lr


def _sample(nav, u, iid, opts):
    """Return (ok, the point just inside volume u's surface) of point iid."""
    seed, tol_ = opts
    o, d, Lr = _point(nav, u, iid, seed)
    h1, te, in1, _, _, _ = nav.first_change_in(u, o, d, F32(0.0), F32(2.0) * Lr)
    h2, tx, _, _, _, _ = nav.first_change_in(u, o, d, te, F32(2.0) * Lr)
    xe = o + te * d
    res = rules.check_step_in(xe) if tol_ is None else tol_
    stp = jnp.minimum(res, 0.5 * jnp.where(h2, tx - te, 2 * res))
    xi = o + (te + stp) * d
    return h1 & in1 & nav.holds_point(u, xi), xi


def _holders(nav, u, xi, pg, kmax):
    """Return the siblings of u (not pieces of its own union) holding xi: up to kmax indices, -1 padded."""
    D = nav.dev()
    n = nav.n

    def visit(i, a):
        hit_ = ((i != u) & (i < n) & ~((pg[u] >= 0) & (pg[jnp.minimum(i, n)] == pg[u]))
                & nav.holds_point(jnp.minimum(i, n - 1), xi))  # fmt: skip
        slot = jnp.argmax(a < 0)
        return jnp.where(hit_ & (a[slot] < 0), a.at[slot].set(i), a)

    return nav.scan(D["parent"][u], xi, visit, jnp.full((kmax,), -1, I32), lambda a: jnp.all(a >= 0))


def _across(nav, u, xi, pg):
    """Return the volume the tree locates xi in when it is neither u nor below it (-1 otherwise)."""
    D = nav.dev()
    n = nav.n
    loc = nav.locate(xi)

    def up(c):
        v_, k_ = c
        return jnp.where(v_ < n, D["parent"][jnp.minimum(v_, n - 1)], v_), k_ + 1

    anc, _ = lax.while_loop(lambda c: (c[0] != u) & (c[0] < n) & (c[1] < MAX_UP), up, (loc, I32(0)))
    return anc, loc, ~((pg[u] >= 0) & (pg[jnp.minimum(loc, n)] == pg[u]))


def _one(nav, pg, opts):
    """Return the per-point test: (ok, protrudes, holders, the across-levels holder)."""
    D = nav.dev()
    n = nav.n
    seed, kmax, tol = opts
    tol_ = None if tol is None else F32(tol)

    def one(u, iid):
        ok, xi = _sample(nav, u, iid, (seed, tol_))
        pv = D["parent"][u]
        out_m = (pv < n) & ~nav.holds_point(jnp.minimum(pv, n - 1), xi)
        hold = _holders(nav, u, xi, pg, kmax)
        anc, loc, other = _across(nav, u, xi, pg)
        cross = ok & (anc != u) & other
        return ok, ok & out_m, jnp.where(ok, hold, -1), jnp.where(cross, loc, -1)

    return one


def _pairs(a, b):
    if len(a):
        return np.unique(np.stack([a, b], 1), axis=0, return_counts=True)
    return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)


def check(nav, n_pts=N_POINTS, n_min=N_MIN, seed=0, kmax=K_MAX, tol=None, backend=None, chunk=CHUNK):  # noqa: PLR0913 -- the check's settings
    """Return Geant4's preconditions tested at points just inside each volume's surface.

    n_pts points per definition (kind x placement), spread over its instances, at least n_min per instance: a volume
    placed 25000 times is not sampled 25000x as densely as one placed once. The pieces of one split union (a
    definition ``<name>|pc<k>`` placed as ``pc<k>``) overlap or touch by construction and are not reported. Returns
    dict(pairs (m, 2) (earlier, later), counts, npts (points per volume), protrude (volumes), n_points, n_tried,
    cross (volume, holder) and cross_counts).
    """
    n = nav.n
    opts = (int(seed), int(kmax), None if tol is None else float(np.float32(tol)))

    def check_point(nv, extra):
        return _one(nv, extra["pg"], opts)

    run = _program(("check", check_point), nav, dict(pg=_piece_groups(nav)), opts, backend)
    _, inv, cnt = np.unique([d_.split("|")[0] for d_ in nav.defs], return_inverse=True, return_counts=True)
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
    cpr, ccnt = _pairs(u_all[cs], cross[cs])
    a = np.repeat(u_all[:, None], kmax, 1)
    sel = hold >= 0
    pa, pb = a[sel], hold[sel]
    pr, cnt = _pairs(np.minimum(pa, pb), np.maximum(pa, pb))
    return dict(pairs=pr, counts=cnt, npts=npts, protrude=np.unique(u_all[prot]), n_points=int(ok.sum()),
                n_tried=len(ok), cross=cpr, cross_counts=ccnt)  # fmt: skip


def _mesh_one(nav, seed, kmax):
    D = nav.dev()
    n = nav.n

    def one(c, iid):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), iid)
        k1, k2 = jax.random.split(key)
        k = D["cstart"][c] + jnp.minimum((jax.random.uniform(k1, (), F32) * D["cn"][c]).astype(I32), D["cn"][c] - 1)
        u = jax.random.uniform(k2, (2,), F32)
        u = jnp.where(u[0] + u[1] > 1, 1 - u, u)
        p = D["V0"][k] + u[0] * (D["V1"][k] - D["V0"][k]) + u[1] * (D["V2"][k] - D["V0"][k])
        mv = D["cmoth"][c]
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

    def mesh_point(nv, extra):
        return _mesh_one(nv, *opts)

    run = _program(("check_meshes", mesh_point), nav, {}, opts, backend)
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


def report(nav, c, m=None):
    """Return the check's summary lines: overlapping pairs (touches apart), protrusions, across levels, meshes.

    A touch is a shared face within the step in, not material: a family (definition pair) whose pairs hold at most
    ``TOUCH`` of every ``TOUCH_PER`` points the two volumes contributed.
    """

    def hd(i):
        return nav.defs[int(i)].split("|")[0]

    fk = [tuple(sorted((hd(a), hd(b)))) for a, b in c["pairs"]]
    num, den = Counter(), Counter()
    for f_, (a, b), k_ in zip(fk, c["pairs"], c["counts"], strict=False):
        num[f_] += int(k_)
        den[f_] += int(c["npts"][a] + c["npts"][b])
    touch = np.array([num[f_] * TOUCH_PER <= TOUCH * den[f_] for f_ in fk], bool).reshape(-1)
    pairs = np.asarray(c["pairs"]).reshape(-1, 2)
    fam = Counter(tuple(sorted((hd(a), hd(b)))) for a, b in pairs[~touch])
    ftc = Counter(tuple(sorted((hd(a), hd(b)))) for a, b in pairs[touch])
    lines = [f"overlaps ({c['n_points']} surface points): {int((~touch).sum())} overlapping sibling pairs (> {TOUCH} "
             f"of every {TOUCH_PER} points of their family; points per pair "
             f"{np.sort(np.asarray(c['counts'])[~touch])[::-1][:SHOWN].tolist()}) {dict(fam.most_common(SHOWN))}; "
             f"{int(touch.sum())} touches {dict(ftc.most_common(4))}; {len(c['protrude'])} protruding volumes "
             f"{dict(Counter(hd(u) for u in c['protrude']).most_common(SHOWN))}"]  # fmt: skip
    cf = Counter()
    for (u_, v_), k_ in zip(c["cross"], c["cross_counts"], strict=False):
        cf[(hd(u_), hd(v_) if v_ < nav.n else "OUT")] += int(k_)
    lines.append(f"overlaps across levels (a point of a volume the tree locates outside it and its daughters): "
                 f"{len(c['cross'])} (volume, holder) pairs, {int(np.sum(c['cross_counts']))} points; "
                 f"{dict(cf.most_common(SHOWN))}")  # fmt: skip
    if m is not None and nav.n_mesh:
        lines.append(f"meshes ({m['n_points']} points): pass-throughs (mesh, volume): points "
                     f"{dict(sorted(((k, hd(v)), x) for (k, v), x in m['pairs'].items())[:SHOWN])}; protruding meshes "
                     f"{m['protrude']}")  # fmt: skip
    return lines
