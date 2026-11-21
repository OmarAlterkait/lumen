# Working on lumen

The entry page for developers. Read this first, then the page you need:

| Page | What it answers |
|---|---|
| [architecture.md](architecture.md) | how lumen is built: layout, data flow, the detector file, the Geant4 rules, gradients, surfaces, readout |
| [decisions.md](decisions.md) | why a choice was made, and what would change it |
| [style.md](style.md) | the numbered coding rules the hooks and `pyproject.toml` cite |
| [validation.md](validation.md) | how the gradient check decides, and the measurements behind the known limits |
| [vocabulary.md](vocabulary.md) | the code's words against the user's words |
| [roadmap.md](roadmap.md) | what is not built yet |
| [../../CHANGELOG.md](../../CHANGELOG.md) | what changed for users, per release |

## Repository map

| Path | What | Shipped (wheel) |
|---|---|---|
| `src/lumen/` | the package; the module map is in architecture.md, "Package layout" | yes |
| `src/lumen/compare/references.json` | lumen's own reference manifest (the overlap toys), each product with its release asset | yes |
| `tests/unit/` | the unit tier (CPU) | no |
| `tests/physics/` | the physics tier (a GPU): toys with closed forms, GPU against CPU, no-recompile, the persistent cache | no |
| `tests/data/` | fixtures the tests read | no |
| `docs/` | user documentation (Diátaxis: tutorial, howto, reference, explanation), the generated validation report, and this directory | no |
| `examples/` | the scripts the tutorial and how-to guides follow | no |
| `tools/ci/gpu.sh` | the GPU part of the checks (GitHub has no GPU) | no |
| `tools/release_assets.py` | packs a manifest's reference products into release assets and records their URL and sha256 | no |
| `tools/g4ref/` | g4ref, the Geant4 reference application, its runner, the reduction into reference products, the MC-gate store script; boundary, the one-interface program of the facet-path references (`tests/data/g4_boundary`) | no |
| `tools/docs/`, `tools/hooks/` | the generated pages' writers; the pre-commit hooks | no |

Where new things go:

| New thing | Where |
|---|---|
| a Geant4 rule (navigation, boundary, optics) | `src/lumen/rules/`, with a test against the Geant4 source line it follows |
| a general import rewrite | `src/lumen/build/import_rules.py` (never a recipe flag: decisions.md, 1) |
| an experiment's detector | its package (lumen-wc, lumen-tpc, or your own): the recipe, value sets, reference manifest and tests; converted with `lumen.convert` (no registry) |
| a medium model | a value passed to the build (`models.py`, `build/library.py`'s model table); `docs/howto/extending.md` |
| a user-visible change | an entry under the next version's heading in `CHANGELOG.md` |
| a decision and its reason | a new entry in `decisions.md`; the code points to it |
| a user error message | the code (the generated section of `docs/reference/errors.md` lists it); a hand-written row there when a user needs the remedy |
| a measured limit | `docs/explanation/known-limits.md` (user words); the measurement in `validation.md` |
| a site path, job script or private input | nowhere in the repository (style rules 9 and 13) |

## Development setup

```bash
git clone https://github.com/OmarAlterkait/lumen.git && cd lumen
python -m venv /path/to/envs/lumen-dev && source /path/to/envs/lumen-dev/bin/activate
pip install -e ".[cuda,dev]"
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

Keep caches off a small HOME: set `LUMEN_CACHE` (see `docs/howto/install.md`) and `PRE_COMMIT_HOME`.

## Tests

| Tier | Command | Needs | Time |
|---|---|---|---|
| unit | `pytest -n 4` (`tests/unit`, the default; `-m "not slow"` as GitHub runs it) | a CPU (`JAX_PLATFORMS=cpu`) | 1.5-5 min per shard of 4 on a 4-core node, without `slow` (measured on an AMD EPYC cluster node; on GitHub not yet) |
| physics | `pytest tests/physics` (or one file at a time); every test is marked `gpu` and skipped without one, except `test_cpu_target.py` | one GPU | 5-10 min per file |
| examples | `python examples/<script>.py` from the repository root | one GPU; `examples/compare_with_geant4.py` also needs a g4ref build (`G4REF`, `tools/g4ref`) | a few minutes each |
| MC gate | `python -m lumen.compare gate --package PKG [NAME ...]` on each family's detectors: the manifest's `mc` entries, their reference products fetched and sha256-checked, converted with its recipes (`lumen.convert("PKG:NAME")`) and compared bit for bit with the reference records, the environment recorded (validation.md, "The MC gate and its re-baseline") | one GPU model, the references (fetched) | minutes per detector |
| comparisons | `python -m lumen.compare overlaps coinc_sib`; the families' `python -m lumen.compare charge|gradients|geometry <names> --package lumen_wc` (and lumen_tpc) | the reference products (fetched into `LUMEN_DATA` on first use), a GPU | per check |

What runs where:

| Where | Runs |
|---|---|
| GitHub, every push and pull request (`.github/workflows/ci.yml`) | lint (ruff, mypy, the site-path check); the unit tier without the tests marked `slow`, in 4 shards on Python 3.10 and 3.13; the photon loop's CPU target |
| a GPU host, before a merge to main and before a tag: `tools/ci/gpu.sh ../lumen-wc ../lumen-tpc` | the physics tier, the `slow` unit tests, the examples, each family's tests and notebooks and its MC gate (bitwise) |
| the families, before a release | the gradient gates on 64 seeds; the comparisons with Geant4, WCSim and LArSoft |

Results are bitwise reproducible on one GPU model only (architecture.md, "Engineering"); a change meant to keep outputs
identical is checked bitwise on the same GPU model as its reference.

## Hooks

`.pre-commit-config.yaml` runs, all locally: ruff format and ruff check, mypy on the host-side modules, the site-path
check (style rule 9: no absolute path outside the repository, no Slurm account) and the privacy guard on staged
content, commit messages and every pushed commit. The guard reads its patterns from the file named by
`LUMEN_PRIVATE_PATTERNS`, kept outside the repository; unset, it does nothing and says so.

## Generated pages

- `docs/validation/report.md`: `python -m lumen.compare report <record dirs> --out docs/validation/report.md`, from the JSON
  records the checks write. Never edit it by hand.
- `docs/reference/api.md`: `python tools/docs/api_reference.py`, from the docstrings (a unit test fails when it is
  out of date).
- The last section of `docs/reference/errors.md`, every message: `python tools/docs/errors_reference.py`, from the raise
  sites (a unit test fails when it is out of date). The sections above it are written by hand.
- The outputs printed in the tutorial and the how-to guides: rerun the example scripts and paste their outputs.

## Release checklist

1. GitHub's CI passes on the release commit, and `tools/ci/gpu.sh ../lumen-wc ../lumen-tpc` passes on a GPU host;
   record both runs (the commit, the job) in the release's notes.
2. Bitwise checks on the five public detectors (the MC gate, driven from lumen-wc and lumen-tpc) where a change is
   meant to keep outputs identical.
3. `CHANGELOG.md`: the new version's heading, with the date; `CITATION.cff`: version and `date-released`; the README's
   status and install.md's pin (`tests/unit/test_release.py` checks that they agree).
4. Re-baseline in this order, when results move: the tallymap tag first, then the families' MC-gate records and
   validation reports, then the tutorial's outputs. Pack each manifest's products (`tools/release_assets.py`) and
   upload them to the release named in their URLs.
5. The printed outputs in the tutorial and the how-to guides match a run of the examples on this version.
6. The validation report regenerated on the released software stack, if the physics changed.
7. `docs/reference/errors.md`: its generated section is current (the unit test), and its hand-written rows still name
   messages the code raises.
8. The privacy guard passes on every commit to be pushed.
9. Push only the squashed `main` and the tag: `git log --format='%ae %ce' origin/main..main` shows only the noreply
   address.
