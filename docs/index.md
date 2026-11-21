# lumen documentation

The documentation follows the Diátaxis layout:

- **Tutorial** (learning by doing): [the first hour](tutorial/first-hour.md), on the toy detector shipped in
  `examples/`.
- **How-to guides** (a task, start to finish): [installing lumen](howto/install.md), GPU or CPU, the caches and
  their variables; [your own detector](howto/your-own-detector.md), from a GDML file to a calibration and a
  comparison with Geant4; [light sources](howto/sources.md), the sources lumen ships (SK, WCTE, ANNIE,
  Hyper-K and IWCD are lumen-wc's, SBND, MicroBooNE, ICARUS and DUNE lumen-tpc's); [extending
  lumen](howto/extending.md), your own source, sensor response or medium model, with its parameters as keys;
  [a detector written in YAML](howto/yaml-detector.md), by hand.
- **Reference**: [the API](reference/api.md), every public function and its signature; [sensor value
  sets](reference/value-sets.md), the format of a sensor model's values; [errors](reference/errors.md), every error
  lumen raises with its fix; [vocabulary](reference/vocabulary.md), the words lumen uses.
- **Explanation** (background): what is a parameter and what is not ([parameters](explanation/parameters.md)),
  measured limits ([known limits](explanation/known-limits.md)) and how the checks decide
  ([validation](explanation/validation.md)). The [validation report](validation/report.md) is generated from the
  checks' records by `python -m lumen.compare report`.
- [`dev/`](dev/README.md): for people working on lumen itself.
- What changed between releases: the [changelog](../CHANGELOG.md).

The tutorial and the how-to guide follow scripts in `examples/` that run as they stand: `examples/first_hour.py` and
`examples/your_own_detector.py`. The extending guide's worked examples are `examples/extending.py` verbatim (a unit
test checks it), each checked against a finite difference by `tests/physics/test_extend.py`.
