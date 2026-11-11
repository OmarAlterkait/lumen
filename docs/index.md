# lumen documentation

The documentation follows the Diátaxis layout:

- **Tutorial** (learning by doing): [the first hour](tutorial/first-hour.md), on the toy detector shipped in
  `examples/`.
- **How-to guides** (a task, start to finish): [your own detector](howto/your-own-detector.md), from a GDML file to
  a calibration and a comparison with Geant4; [extending lumen](howto/extending.md), your own source, sensor
  response, medium or surface model, with its parameters as keys.
- **Reference**: [errors](reference/errors.md), every error lumen raises with its fix; [vocabulary](reference/vocabulary.md),
  the words lumen uses.
- **Explanation** (background): what is a parameter and what is not ([parameters](explanation/parameters.md)),
  measured limits ([known limits](explanation/known-limits.md)) and how the checks decide
  ([validation](explanation/validation.md)). The [validation report](validation/report.md) is generated from the
  checks' records by `lumen validate report`.
- `dev/`: for people working on lumen itself.

Every code block of the tutorial and the how-to guide is taken from a script in `examples/` that runs as it stands:
`examples/first_hour.py`, `examples/your_own_detector.py` and `examples/compare_with_geant4.py`; the extending guide's
worked examples are `examples/extending.py`, each checked AD against FD by `tests/physics/test_extend.py`.
