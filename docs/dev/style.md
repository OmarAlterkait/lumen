# Style rules

Numbered so that `pyproject.toml`, the pre-commit hooks and review comments can cite them ("style rule 8"). Keep the
numbers stable: add new rules at the end.

| # | Rule | Enforced by |
|---|---|---|
| 1 | Short uppercase math names (`R`, `T`, `Es`) only inside a function whose docstring has a `Notation` line | review (ruff's N806 is off) |
| 2 | Functions have at most 50 statements, 12 branches, McCabe complexity 12 and 6 positional arguments; a `# noqa: PLR0913` says why | ruff (`pyproject.toml`) |
| 3 | Records are NamedTuples or frozen dataclasses, built by keyword | review |
| 4 | Protocols, never `getattr` discovery | review |
| 5 | numpydoc docstrings with units: what it is, its units, the Geant4 source it follows (class::method or file:line), what each non-obvious argument means | ruff `D` (numpy convention), review |
| 6 | Comments say why, never when; no port history, plan codes, job IDs or out-of-repo file names | review, the jargon hook (rule 10) |
| 7 | Named constants carry their unit and source | review |
| 8 | Only `lumen._env` reads the environment | ruff `TID251` (`os.environ`, `os.getenv` banned elsewhere) |
| 9 | No absolute site paths in `src/`, `tests/`, `refs/` or `tools/` | the `no-site-paths` hook |
| 10 | Text a user reads uses plain physics words, never internal terms or plan codes | the `jargon` hook (`tools/hooks/jargon.py`) |
| 11 | A module docstring is at most about 15 lines: purpose, a usage example, units and the Geant4 source followed, and a pointer to `architecture.md` (by section name) or `decisions.md` for rules and rationale; long rule tables that are the module's contract may stay, and the docs link to them instead of copying them | review |
| 12 | Units are m, ns, nm and eV everywhere; unit suffixes appear only in file field names | review |
| 13 | Nothing private enters the repository | the privacy-guard hooks (`LUMEN_PRIVATE_PATTERNS`, kept outside the repository) |
