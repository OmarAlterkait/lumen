"""The CPU tier: tallymap's kernel backend compiled for the CPU (C++/OpenMP), on a toy, forward and jvp.

``backend="cpu"`` runs tallymap's CPU target (on a GPU machine lumen puts the run on the CPU and its outputs back).
Its charge and derivatives on the Lambertian sensor sphere (a drawn reflection per bounce, so a real Monte Carlo) are
compared with the GPU's on a GPU machine, and with tallymap's plain-JAX backend (``backend="xla"``) without one: the
same seeds and rows on both, so the seeds are paired (``stats.mean_test``, paired, at the suites' level): they agree
bit for bit or within the spread of their per-seed differences (float32 rounding that turns a decision). Runs
without a GPU
(``JAX_PLATFORMS=cpu pytest tests/physics/test_cpu_target.py``); needs a C++ compiler with OpenMP, else skipped. The
overlap check's "cpu" runs there too, and finds what the reference finds.
"""

import os
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_toys import charge, families, rows_at
from toys import GPU

from lumen import run as runner
from lumen.build import placements
from lumen.tools import stats

K, N = 16, 4096  # seeds, photons per seed


def _openmp() -> bool:
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        return False
    r = subprocess.run([cxx, "-fopenmp", "-x", "c++", "-", "-o", os.devnull], input="int main() { return 0; }\n",
                       capture_output=True, text=True)  # fmt: skip
    return r.returncode == 0


needs_openmp = pytest.mark.skipif(not _openmp(), reason="the CPU target needs a C++ compiler with OpenMP (g++)")


def _forward_and_jvp(s, backend):
    """Per-seed charge per photon and its jvp along R and along the absorption length; the primal's devices."""
    p0, fams = families(s, ".R[", "Liquid:absorption")
    q = charge(s, p0, fams, rows_at(K * N), seeds=K, backend=backend)
    th = jnp.zeros(2, jnp.float32)
    v = q(th)
    out = [np.asarray(v, np.float64)]
    out += [np.asarray(jax.jvp(q, (th,), (th.at[j].set(1.0),))[1], np.float64) for j in range(2)]
    return out, v.devices()


def _ran_on_the_cpu():
    return runner.tm.history()["kernels"][-1]["target"] == "cpu"


@needs_openmp
def test_the_cpu_target_agrees_with_the_reference_within_mc_error(toy):
    s = toy("sphere", r=0.5, finish="ground", sigma_alpha=0.1, absorption=2.0)
    cpu, where = _forward_and_jvp(s, "cpu")  # on a GPU machine too: lumen moves the run to the CPU
    assert _ran_on_the_cpu()
    assert where == {jax.devices()[0]}  # the outputs come back to JAX's default device
    ref, _ = _forward_and_jvp(s, "auto" if GPU else "xla")  # the GPU, else tallymap's plain-JAX backend
    for name, a, b in zip(("Q", "dQ/dR", "dQ/dabsorption"), cpu, ref, strict=True):
        t = stats.mean_test(a[:, None], b[:, None], paired=True, level=stats.ALPHA / stats.SUITE_TESTS)
        assert np.array_equal(a, b) or t["passed"] is True, (name, a.mean(), b.mean(), t["stat"], t["passed"])
        assert np.all(np.isfinite(a)) and abs(a.mean()) > 0, name


@needs_openmp
def test_the_overlap_check_runs_on_the_cpu_target(toy):
    nav = toy("sphere", r=0.5).model.geometry.nav
    c = placements.check(nav, n_pts=128, backend="cpu")
    assert _ran_on_the_cpu()
    ref = placements.check(nav, n_pts=128, backend="auto" if GPU else "xla")
    assert all(np.array_equal(np.asarray(c[k]), np.asarray(ref[k])) for k in ref), (c, ref)
