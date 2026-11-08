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
cluster/s3df/make_env.sh <ENV_DIR>    # a venv inside develop.sif, outside HOME
source cluster/s3df/env.sh <ENV_DIR>  # variables, caches, and `run` on PATH
run python -m pytest                                               # inside the container
```

## Documentation

- [Glossary](docs/explanation/glossary.md): the words lumen uses.
- [Developer vocabulary](docs/dev/vocabulary.md): how those words map to terms in the earlier code.

## Licence

Not chosen yet; see [LICENSE](LICENSE).
