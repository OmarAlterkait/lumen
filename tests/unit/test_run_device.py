"""A run that expects the GPU never falls back to the CPU silently (lumen.run.DeviceError)."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from lumen import run as runner


def test_backend_cuda_refuses_a_cpu_jax(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(runner.DeviceError, match="expects the GPU"):
        runner.run(None, None, None, None, None, backend="cuda")


@pytest.mark.parametrize(("env", "raises"), [({"JAX_PLATFORMS": "cuda"}, True), ({"JAX_PLATFORMS": "cpu"}, False),
                                             ({"SLURM_JOB_GPUS": "0"}, True), ({"SLURM_GPUS_ON_NODE": "1"}, True),
                                             ({}, False)])  # fmt: skip
def test_backend_auto_refuses_a_cpu_jax_only_where_a_gpu_is_expected(monkeypatch, env, raises):
    for v in ("JAX_PLATFORMS", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        monkeypatch.delenv(v, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    if raises:
        with pytest.raises(runner.DeviceError):
            runner.check_device("auto")
    else:
        runner.check_device("auto")
    runner.check_device("cpu")  # an explicit CPU run never expects the GPU
    runner.check_device("xla")


def test_backend_cpu_moves_the_run_to_the_cpu_where_jax_would_use_the_gpu(monkeypatch):
    """backend="cpu" never refuses: where JAX places computations on a GPU, the run is put on the CPU (on_device)."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    runner.check_device("cpu")
    seen = {}

    def f(x):
        seen["device"] = jax.config.jax_default_device
        return x + 1

    out = runner.on_device("cpu", f, jnp.ones(2))
    assert seen["device"].platform == "cpu" and float(out.sum()) == 4.0
    seen.clear()
    runner.on_device("auto", f, jnp.ones(2))
    assert seen["device"] is None  # other backends run where JAX places them


def test_an_unknown_backend_name_is_refused():
    with pytest.raises(ValueError, match="backend must be one of"):
        runner.resolve_backend("vmap")


def test_without_a_cpp_compiler_the_cpu_target_says_what_to_install(tmp_path):
    """No C++ compiler with OpenMP: the CPU target raises tallymap's UnsupportedError naming g++, never falls back."""
    code = (
        "import jax.numpy as jnp\n"
        "from lumen import run\n"
        "name = run.resolve_backend('cpu')[0]\n"
        "f = run.vmap_runner(lambda tl, x: tl.add('z', 0, x), {'z': run.sum_output(1)}, name)\n"
        "f(jnp.ones(4, jnp.float32))\n"
    )
    env = dict(os.environ, CXX=str(tmp_path / "no-compiler"), JAX_PLATFORMS="cpu", TALLYMAP_CACHE=str(tmp_path))  # noqa: TID251 -- the child's environment
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "UnsupportedError" in r.stderr and "install g++" in r.stderr, r.stderr[-3000:]
