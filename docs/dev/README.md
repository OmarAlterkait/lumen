# Working on lumen

The entry page for developers and coding agents. Read this first, then the page you need:

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

| Path | What | Shipped |
|---|---|---|
| `src/lumen/` | the package; the module map is in architecture.md, "Package layout" | wheel and sdist |
| `src/lumen/detectors/`, `src/lumen/sensors/data/` | the shipped recipes, optics tables, value sets and the reference manifest | yes |
| `tests/unit/` | the unit tier (CPU) | sdist |
| `tests/physics/` | the physics tier (a GPU): toys with closed forms, GPU against CPU, no-recompile, the persistent cache | sdist |
| `tests/data/` | fixtures the tests read | sdist |
| `docs/` | user documentation (Diátaxis: tutorial, howto, reference, explanation), the generated validation report, and this directory | sdist |
| `examples/` | the scripts every code block in the tutorial and how-to guides comes from | no |
| `refs/` | how the reference products were made (g4ref runs, reductions, patches) | no |
| `tools/hooks/` | the pre-commit hooks | no |
| `tools/readout/transcribe.py` | regenerates the shipped value sets in `src/lumen/sensors/data/` from WCSim and sbndcode sources | no |

Where new things go:

| New thing | Where |
|---|---|
| a Geant4 rule (navigation, boundary, optics) | `src/lumen/rules/`, with a test against the Geant4 source line it follows |
| a general import rewrite | `src/lumen/build/import_rules.py` (never a recipe flag: decisions.md, 1) |
| a detector shipped with lumen | a recipe in `src/lumen/detectors/`, its entry point in `pyproject.toml` if it is a separate package |
| an optics or sensor model | registered by name (`models.py`, `plugins.py`); `docs/howto/extending.md` |
| a user-visible change | an entry under "Unreleased" in `CHANGELOG.md` |
| a decision and its reason | a new entry in `decisions.md`; the code points to it |
| a user error message | the code, and a row in `docs/reference/errors.md` |
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
| unit | `pytest tests/unit -n 4` | a CPU (`JAX_PLATFORMS=cpu`) | a few minutes on 4 workers |
| physics | `pytest tests/physics` (or one file at a time) | one GPU | 5-10 min per file |
| examples | `python examples/<script>.py` from the repository root | one GPU; `examples/compare_with_geant4.py` also needs a g4ref build (`G4REF`) | a few minutes each |
| validation | `lumen validate charge|gradients|geometry|overlaps <names>` | the reference store (`LUMEN_DATA`), a GPU | per check |

Results are bitwise reproducible on one GPU model only (architecture.md, "Engineering"); a change meant to keep outputs
identical is checked bitwise on the same GPU model as its reference.

## Hooks

`.pre-commit-config.yaml` runs, all locally: ruff format and ruff check, mypy on the host-side modules, the jargon
check on user-facing text (style rule 10), the site-path check (style rule 9) and the privacy guard on staged
content, commit messages and every pushed commit. The guard reads its patterns from the file named by
`LUMEN_PRIVATE_PATTERNS`, kept outside the repository; unset, it does nothing and says so.

## Generated pages

- `docs/validation/report.md`: `lumen validate report <record dirs> --out docs/validation/report.md`, from the JSON
  records the checks write. Never edit it by hand.
- The outputs printed in the tutorial and the how-to guides: rerun the example scripts and paste their outputs.

## Release checklist

1. The unit and physics tiers pass; the examples run.
2. Bitwise checks on the shipped detectors where a change is meant to keep outputs identical.
3. `CHANGELOG.md`: move "Unreleased" under the new version, with the date.
4. `CITATION.cff`: version and `date-released`.
5. The printed outputs in the tutorial and the how-to guides match a run of the examples on this version.
6. The validation report regenerated on the released software stack, if the physics changed.
7. `docs/reference/errors.md` matches the messages the code raises.
8. The privacy guard passes on every commit to be pushed.
