"""The structural rules every detector structure meets, whatever it was read from (lumen.format.check)."""

import dataclasses

import numpy as np
import pytest
from test_importer import GDML, write

from lumen.format import schema
from lumen.format.check import StructureError, check
from lumen.gdml import importer


@pytest.fixture
def df(tmp_path):
    d = importer.read_gdml(write(tmp_path, GDML))
    return dataclasses.replace(d, detector_volume="TankLV")


def _material(df, **kw):
    return dataclasses.replace(
        df, materials={**df.materials, "Water": dataclasses.replace(df.materials["Water"], **kw)}
    )


def test_a_read_gdml_structure_passes_and_decisions_are_checked_apart(df):
    check(df)
    raw = dataclasses.replace(df, detector_volume="Nope")
    check(raw, decisions=False)  # a GDML read before its recipe decides
    with pytest.raises(StructureError, match="detector volume 'Nope'"):
        check(raw)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: dataclasses.replace(d, world="Hall"), "the world 'Hall' is not a volume"),
        (lambda d: dataclasses.replace(d, placements=d.placements[:-1]), "daughter 2 names no placement"),
        (
            lambda d: dataclasses.replace(
                d, placements=(dataclasses.replace(d.placements[0], mother="PmtLV"), *d.placements[1:])
            ),
            "is not listed in its mother",
        ),
        (
            lambda d: dataclasses.replace(
                d, solids={**d.solids, "Tank": dataclasses.replace(d.solids["Tank"], params={"x": 1.0, "r": 2.0})}
            ),
            r"unknown attributes \['r'\]",
        ),
        (lambda d: _material(d, composition=schema.Composition(1.0)), "names no composition"),
        (
            lambda d: _material(
                d, properties={"RINDEX": schema.Property("RINDEX", np.array([1.3, 1.4]), energy=np.array([3.0, 2.0]))}
            ),
            "'Water'.RINDEX: energies must be strictly ascending",
        ),
        (
            lambda d: dataclasses.replace(
                d, skins=(*d.skins, schema.Skin("S2", surface="Liner", volume=d.skins[0].volume))
            ),
            "two skin surfaces",
        ),
        (
            lambda d: dataclasses.replace(d, borders=(*d.borders, dataclasses.replace(d.borders[0], name="again"))),
            "has a border already",
        ),
        (
            lambda d: dataclasses.replace(d, skins=(dataclasses.replace(d.skins[0], surface="Paint"),)),
            "refers to 'Paint', which is not a surface",
        ),
    ],
)
def test_broken_rules_are_refused_with_their_reason(df, change, match):
    with pytest.raises(StructureError, match=match):
        check(change(df))


def test_a_recipe_skin_may_replace_the_source_s_but_not_another_recipe_skin(df):
    lv = df.skins[0].volume
    recipe = schema.Skin("mine", surface="Liner", volume=lv, origin="recipe")
    check(dataclasses.replace(df, skins=(*df.skins, recipe)))
    with pytest.raises(StructureError, match="two skin surfaces"):
        check(dataclasses.replace(df, skins=(*df.skins, recipe, dataclasses.replace(recipe, name="again"))))


def test_the_format_imports_nothing_above_it():
    """lumen.format is the bottom layer: its modules import only lumen.format and lumen.units (lazily or not)."""
    import ast
    from pathlib import Path

    import lumen.format

    bad = []
    for f in sorted(Path(lumen.format.__file__).parent.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):  # "from lumen import units" names lumen.units
                mod = node.module or ""
                names += [f"{mod}.{a.name}" for a in node.names] if mod == "lumen" else [mod]
            bad += [f"{f.name}: {n}" for n in names if n.startswith("lumen") and n.split(".")[:2]
                    not in (["lumen", "format"], ["lumen", "units"])]  # fmt: skip
    assert not bad, bad


def test_an_omitted_angular_range_is_geant4s_zero_in_gdml_and_the_whole_solid_in_yaml(df):
    """G4GDMLReadSolids reads an omitted deltaphi as 0, where G4Tubs aborts; a YAML or builder solid omitting it is
    the whole solid, stored explicitly (lumen.format.solids.whole)."""
    from lumen.format import solids

    tube = df.solids["Pmt"]
    open_tube = dataclasses.replace(tube, params={k: v for k, v in tube.params.items() if k != "deltaphi"})
    with pytest.raises(StructureError, match=r"deltaphi 0.0 \(omitted: 0\)"):
        check(dataclasses.replace(df, solids={**df.solids, "Pmt": open_tube}))
    full = solids.whole("tube", {"rmax": 0.1, "z": 0.2})
    assert full == {"rmax": 0.1, "z": 0.2, "startphi": 0.0, "deltaphi": 2 * np.pi}
    assert solids.whole("sphere", {"rmax": 1.0})["deltatheta"] == np.pi
    assert solids.whole("tube", {"deltaphi": 1.0})["deltaphi"] == 1.0
    assert solids.whole("polycone", {}) == {"startphi": 0.0, "deltaphi": 2 * np.pi}  # Geant4's 0 is also the full turn
