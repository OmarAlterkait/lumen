"""Comparisons against reference simulations: the checks, their records and the report.

A check (``charge``, ``gradients``, ``geometry``, ``overlaps``) is named by an entry of a reference manifest
(``lumen.compare.references``: lumen's own lists the overlap toys; a detector package ships its own and passes its path)
and writes JSON records (``lumen.compare.records``) that the report is made from (``lumen.compare.report``):

    python -m lumen.compare charge sk --references MANIFEST.json [--detector-file sk.lumen] [--out DIR]

The command is the interface (``python -m lumen.compare``, ``lumen.compare.cli``); the functions behind it are
validation tooling and may change.
"""

__all__: list = []
