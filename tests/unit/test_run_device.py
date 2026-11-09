"""A run that expects the GPU never falls back to the CPU silently (lumen.run.DeviceError)."""

import jax
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
