"""A small table of common elements, so a detector written by hand (YAML, the builder) need not define them.

A material's component that names no element or material of the detector, but one of these symbols, is that element:
its symbol, Z and standard atomic weight. The weights are IUPAC's abridged standard atomic weights (CIAAW, "Standard
atomic weights of the elements 2021", Pure Appl. Chem. 94 (2022) 573), to five significant figures where the
abridged value has more; a detector that needs other values defines the element itself.

    df = with_elements(df)     # every component naming one of ELEMENTS, and no record, gets its Element
"""

from __future__ import annotations

import dataclasses

from lumen.format import schema

# symbol: (Z, standard atomic weight g/mol, IUPAC 2021 abridged)
ELEMENTS = {
    "H": (1, 1.008), "Li": (3, 6.94), "Be": (4, 9.0122), "B": (5, 10.81), "C": (6, 12.011), "N": (7, 14.007),
    "O": (8, 15.999), "F": (9, 18.998), "Ne": (10, 20.180), "Na": (11, 22.990), "Mg": (12, 24.305),
    "Al": (13, 26.982), "Si": (14, 28.085), "P": (15, 30.974), "S": (16, 32.06), "Cl": (17, 35.45),
    "Ar": (18, 39.95), "K": (19, 39.098), "Ca": (20, 40.078), "Ti": (22, 47.867), "Cr": (24, 51.996),
    "Mn": (25, 54.938), "Fe": (26, 55.845), "Ni": (28, 58.693), "Cu": (29, 63.546), "Zn": (30, 65.38),
    "Ge": (32, 72.630), "Br": (35, 79.904), "Kr": (36, 83.798), "Sn": (50, 118.71), "I": (53, 126.90),
    "Xe": (54, 131.29), "Cs": (55, 132.91), "Gd": (64, 157.25), "W": (74, 183.84), "Pb": (82, 207.2),
}  # fmt: skip


def element(symbol: str) -> schema.Element:
    """Return the Element of a symbol of ``ELEMENTS`` (named by its symbol)."""
    z, a = ELEMENTS[symbol]
    return schema.Element(symbol, formula=symbol, z=float(z), molar_mass_g_mol=a)


def with_elements(df: schema.DetectorFile) -> schema.DetectorFile:
    """Return the detector file with an Element for every material component that names a symbol of ``ELEMENTS`` and
    no element or material of the detector (in the order the materials name them), and each component's kind as
    Geant4 resolves a GDML ``fraction``: an element of that name first, else a material (G4GDMLReadMaterials::
    MixtureRead); an element's components are isotopes."""  # noqa: D205, D209
    known = set(df.elements) | set(df.materials)
    new = {c.ref: element(c.ref) for m in df.materials.values() for c in m.composition.components
           if c.ref not in known and c.ref in ELEMENTS}  # fmt: skip
    els = {**df.elements, **new}

    def kinded(c):
        if c.kind:
            return c
        return dataclasses.replace(c, kind="element" if c.ref in els or c.ref not in df.materials else "material")

    mats = {n: dataclasses.replace(m, composition=dataclasses.replace(m.composition, components=tuple(
        kinded(c) for c in m.composition.components))) for n, m in df.materials.items()}  # fmt: skip
    els = {n: dataclasses.replace(e, isotopes=tuple(dataclasses.replace(c, kind="isotope") for c in e.isotopes))
           for n, e in els.items()}  # fmt: skip
    return dataclasses.replace(df, elements=els, materials=mats)


__all__ = ["ELEMENTS", "element", "with_elements"]
