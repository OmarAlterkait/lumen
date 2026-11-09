"""Registries: detectors, sensor models and optics models, by name, from lumen and from installed packages.

Each registry maps ``distribution:name`` (or a bare built-in name) to an object. lumen's own entries are registered
in the package; other packages add theirs through the entry-point groups ``lumen.detectors``,
``lumen.sensor_models`` and ``lumen.optics_models``. An unknown name fails closed, naming what is registered.

``sensor_models`` holds the after-tracking sensor models (detection, electronics, hit coordinates); how a detecting
surface behaves during tracking is optics (``lumen.rules.optics.SENSOR_SURFACES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata

GROUPS = ("lumen.detectors", "lumen.sensor_models", "lumen.optics_models")
BUILTINS = {"lumen.sensor_models": "lumen.sensors"}  # the module that registers lumen's own entries of a group


class NotRegistered(KeyError):  # noqa: N818 (the name of the documented error catalogue)
    """A name no registry entry or installed package provides."""


@dataclass(frozen=True)
class DetectorPackage:
    """What a registered detector provides: its recipe, the dataset ids of its sources, its validation references.

    ``source_dataset`` is the GDML's path under ``LUMEN_DATA``; ``inputs`` the recipe's extra inputs, likewise;
    ``references`` the path of its reference manifest (``lumen.validate.references``; None: lumen's own lists it).
    """

    name: str
    recipe: object
    source_dataset: str
    inputs: dict = field(default_factory=dict)
    references: str | None = None


class Registry:
    """One registry: built-in entries plus the installed packages' entry points of its group, loaded on first use."""

    def __init__(self, group: str):
        self.group = group
        self._entries: dict = {}
        self._loaded = False

    def register(self, name: str, obj) -> None:
        """Register an object; a name registered twice is refused."""
        if name in self._entries and self._entries[name] is not obj:
            raise ValueError(f"{self.group}: {name!r} is registered already")
        self._entries[name] = obj

    def _load_entry_points(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.group in BUILTINS:
            import_module(BUILTINS[self.group])
        for ep in metadata.entry_points(group=self.group):
            if ep.name not in self._entries:
                self._entries[ep.name] = ep

    def get(self, name: str):
        """Return the object registered under ``name``."""
        self._load_entry_points()
        if name not in self._entries:
            raise NotRegistered(f"{self.group}: no entry {name!r}; registered: {sorted(self._entries)}. "
                                "Install the package that provides it.")  # fmt: skip
        obj = self._entries[name]
        if isinstance(obj, metadata.EntryPoint):
            obj = self._entries[name] = obj.load()
        return obj

    def names(self) -> list[str]:
        """Return the registered names."""
        self._load_entry_points()
        return sorted(self._entries)


detectors = Registry("lumen.detectors")
sensor_models = Registry("lumen.sensor_models")
optics_models = Registry("lumen.optics_models")
