"""The film lowering's fail-closed check: films must face the axes the recipe expects (SBND: +-x)."""

import types

import numpy as np
import pytest

from lumen.build import legacy_library
from lumen.build.tree import PV, Tree
from lumen.format import schema


def film_case(rotations, expected=([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])):
    settings = dict(
        film_material="TPB",
        front_material="LAr",
        backing="foil",
        thickness_m=2.3e-8,
        local_axis=[1.0, 0.0, 0.0],
        expected_axes=[list(a) for a in expected],
    )
    film = schema.Surface("cpa_tpb", "", "", "", 0.0, {}, schema.OpticsModel("lumen:wls_film", 1, settings, "recipe"))
    df = types.SimpleNamespace(skins=(schema.Skin("cpa_tpb", "cpa_tpb", "Foil", "recipe"),), surfaces={"cpa_tpb": film})
    pvs = [PV("w", "World", "LAr", None)] + [PV(f"w/f{i}", "Foil", "Steel", "w") for i in range(len(rotations))]
    insts = [types.SimpleNamespace(pv=f"w/f{i}", R=np.asarray(r, float)) for i, r in enumerate(rotations)]
    return df, Tree(tuple(pvs)), insts


FLIP = np.diag([-1.0, 1.0, -1.0])  # 180 degrees about y: the film faces -x
TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 degrees about z: the film faces +y


def test_films_facing_the_expected_axes_are_split():
    tree, films = legacy_library.split_films(*film_case([np.eye(3), FLIP]))
    assert sorted(tuple(ax) for ax in tree.film_axes.values()) == [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert len(films) == 2 and all(lv.startswith("Foil|film") for lv in films)


@pytest.mark.parametrize("rotations", [[np.eye(3), TURN], [np.eye(3)], [np.eye(3), FLIP, TURN]])
def test_a_film_facing_another_axis_is_refused(rotations):
    with pytest.raises(legacy_library.LibraryError, match="expects"):
        legacy_library.split_films(*film_case(rotations))
