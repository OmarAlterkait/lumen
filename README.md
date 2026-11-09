# lumen

A differentiable optical-photon simulation for neutrino detectors, written in JAX.

lumen traces optical photons through a detector imported from GDML with Geant4's optical physics, records the light
reaching each sensor, and gives derivatives of what the sensors see with respect to every parameter except the
geometry. Its photon loop is [tallymap](#install).

**Status: under construction.** This repository is being built step by step from an earlier code. Nothing is
released, and the interfaces will change.

## Install

lumen is not published, and neither is tallymap, the photon-loop engine it needs. Today it runs only on S3DF:

```bash
source cluster/s3df/env.sh                                         # the group's env (jax 0.6.2), caches, `run` on PATH
cluster/s3df/make_env.sh <ENV_DIR>    # or your own copy: the same lock, outside HOME
run python -m pytest tests/unit -n 4                               # tier (a), inside the container: 4 workers
```

## Documentation

- [Glossary](docs/explanation/glossary.md): the words lumen uses.
- [Developer vocabulary](docs/dev/vocabulary.md): how those words map to terms in the earlier code.

## Licence

Not chosen yet; see [LICENSE](LICENSE).
