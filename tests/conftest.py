"""Both tiers' one GPU memory setting, made before anything imports jax.

XLA preallocates a fraction of the GPU (``XLA_PYTHON_CLIENT_MEM_FRACTION``, 0.75 by default) when its backend starts.
A test that starts a second process on the same GPU (``test_persistent_cache``'s three processes) or needs more than
the caller's fraction left (the Mie forward / reverse test) then runs out of memory, depending on the caller's
setting. Without preallocation every process takes only what it allocates, so those tests pass whatever the caller
set; the child processes inherit it.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # noqa: TID251 -- the test process's XLA setting, not lumen's
