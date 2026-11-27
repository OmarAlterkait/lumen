"""Medium models of your own: values, passed to the build, with declared parameters that become keys.

    @lumen.medium_model("mylab:water_hg", params={"g": Param("", (-0.99, 0.99), default=0.9)})
    def water_hg(material, base):                 # base: Geant4's reading of the material (lumen.media.MediumSpec)
        return base.with_channel("rayleigh", lobe=lumen.media.henyey_greenstein("g"))

    det = lumen.load("tank.lumen", parts={"materials/Water": water_hg})
    det.params()["materials/Water/g"]             # 0.9: calibratable, differentiable (reweight)

    det = lumen.load("tank.lumen", parts={"materials/Water/rayleigh": lobe})   # a part: one channel's phase function
    det.params()["materials/Water/rayleigh/a"]    # its parameter, under its channel (lumen.media.phase_function)

A model is a value (``MediumModel``: its name, its function and its declared parameters); nothing is registered. The
build reads each material's optics model by name from its model table: lumen's own (``lumen.build.library.MEDIA``)
and the models passed to it (``lumen.load(..., parts=)``, which also binds them to their records without editing the
file). The file records the name; it never holds code, and lumen never imports code a file names: a name the table
does not hold is refused, and names of lumen's own (``lumen:...``, ``geant4``) are not a model of yours.

Parameters: each declared name is the key ``materials/<M>/<name>`` (a part's: ``materials/<M>/<channel>/<name>``), its
value the record's property of that name when the file has one (a material's model only), else the declared default.
They are scalars. The model reads them by name (a lobe's ``params``); the build maps them as direct reads, so
``pack`` makes them traced and differentiable (reweighting: the draws stay, the weights move).

What a model can return: a ``MediumSpec`` (Geant4's channels with lobes and laws swapped or added). A surface is one of
lumen's surface types, read by lumen's own surface model (``geant4``): every surface row is a row of one parameter
table read by one kernel (``lumen.surfaces.types``), so there are no surface models of your own. Optics that vary
with position are not open.

Checks at first build, per model and lobe: a medium lobe's sampler against its density (a chi^2), at most 3 uniforms
per lobe.
"""

from __future__ import annotations

import dataclasses
import functools
import weakref
from collections.abc import Callable

import numpy as np

from lumen._core.errors import ModelError
from lumen.params import Param

# the checks a model's code passes at its first build (``lumen.build.library``, ``lumen._core.checks``), by
# (model, lobe name)
# a MediumModel of your own -> the lobe names already checked for it (``lumen.build.library``):
# weak keys, so a model a caller drops is not held for the process's life
_CHECKED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@dataclasses.dataclass(frozen=True, eq=False)
class MediumModel:
    """A medium model: its name (``distribution:name``), its function ``fn(material, base) -> MediumSpec`` and its
    declared parameters ((name, Param), ...)."""  # noqa: D205, D209

    name: str
    fn: Callable
    declared: tuple = ()
    part: tuple | None = None  # (channel, lobe): a part, one channel's phase function (``lobe_model``)

    def values(self, rec) -> dict:
        """Return {declared name: value} for a record: its property of that name when present, else the default."""
        out = {}
        for k, p in self.declared:
            prop = rec.properties.get(k) if self.part is None else None  # a part's: its default (no file property)
            if prop is not None and prop.energy is not None:
                raise ModelError(f"model {self.name!r}: {rec.name!r} gives {k} as a table; a declared parameter is one "
                                 "number (a model's parameters cannot be wavelength tables)")  # fmt: skip
            out[k] = float(np.asarray(prop.value)) if prop is not None else p.default
        return out

    def keys(self, rec) -> dict:
        """Return {parameter: (key, Source column, note)} of a record: a declared name's column is "constant" when the
        file has it, else "default"."""  # noqa: D205, D209
        at = f"materials/{rec.name}/" + ("" if self.part is None else f"{self.part[0]}/")
        own = rec.properties if self.part is None else {}
        return {k: (at + k, "constant" if k in own else "default", f"declared by {self.name}")
                for k, _ in self.declared}  # fmt: skip


def medium_model(name: str, params: dict | None = None):
    """Return a decorator making ``f(material, base) -> MediumSpec`` the MediumModel ``name`` (a value)."""
    if ":" not in name or name.startswith("lumen:"):
        raise ValueError(f"optics model {name!r}: a 'distribution:name', not one of lumen's own")
    for k, p in (params or {}).items():
        if not isinstance(p, Param) or p.moves or p.shape != () or p.default is None:
            raise TypeError(f"optics model {name!r}: {k!r} must be a scalar lumen.Param with a default")
        if "/" in k:
            raise ValueError(f"optics model {name!r}: parameter name {k!r} (no '/')")
    return lambda fn: MediumModel(name, fn, tuple(sorted((params or {}).items())))


@functools.cache
def lobe_model(channel: str, lobe) -> MediumModel:
    """Return the medium model "Geant4's reading, channel ``channel``'s lobe replaced by ``lobe``": a part.

    ``lobe``: a phase function with its declared parameters (``lumen.media.phase_function``), given as
    ``load(parts={"materials/<M>/<channel>": lobe})``; its keys are ``materials/<M>/<channel>/<name>``. One model per
    (channel, lobe): materials given one lobe share it.
    """
    if lobe.params and not lobe.declared:
        raise ModelError(f"lobe {lobe.name!r}: its parameters {lobe.params} are not declared "
                         "(lumen.media.phase_function)")  # fmt: skip
    if "/" in channel or not channel:
        raise ValueError(f"a part's channel is one name, not {channel!r}")
    fn = lambda rec, base: base.with_channel(channel, lobe=lobe)  # noqa: E731
    return MediumModel(f"{lobe.name}@{channel}", fn, lobe.declared, (channel, lobe))


def model_table(models=()) -> dict:
    """Return {name: MediumModel} of the models passed to a build (an iterable, or a mapping's values), refusing a
    name two models share."""  # noqa: D205, D209
    out: dict = {}
    for m in models.values() if isinstance(models, dict) else models:
        if not isinstance(m, MediumModel):
            raise TypeError(f"a model is a MediumModel (lumen.medium_model), not {type(m).__name__}")
        if out.setdefault(m.name, m) is not m:
            raise ValueError(f"optics model {m.name!r}: two models of one name")
    return out


__all__ = ["MediumModel", "lobe_model", "medium_model", "model_table"]
