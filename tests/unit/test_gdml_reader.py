"""The GDML reader (lumen.gdml.reader): the expression evaluator, lengths in metres, fail-closed reading, and the
normalisation of exporter defects (denormal numbers, identical re-definitions, matrices not a multiple of coldim; a
conflicting re-definition is an error). Ported from the baseline's tests/test_scene_gdml.py."""

import math

import numpy as np
import pytest

from lumen.gdml import reader

HEAD = (
    '<?xml version="1.0"?>\n<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:noNamespaceSchemaLocation="gdml.xsd">\n'
)
WORLD = """ <solids>
  <box name="wbox" x="1000" y="1000" z="1000" lunit="mm"/>
  {solids}
 </solids>
 <structure>
  {volumes}
  <volume name="World"><materialref ref="Air"/><solidref ref="wbox"/>{physvols}</volume>
 </structure>
 <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


def _gdml(tmp_path, define="", solids="", volumes="", physvols="", name="t.gdml"):
    p = tmp_path / name
    p.write_text(
        HEAD + f" <define>\n{define}\n </define>\n" + WORLD.format(solids=solids, volumes=volumes, physvols=physvols)
    )
    return str(p)


def test_evaluator_units_constants_matrices_functions():
    ev = reader.Evaluator()
    ev.define("b", "2*a")  # defined before a: resolved lazily
    ev.define("a", "3")
    ev.define("q", "1.5", "cm")
    ev.matrices["m"] = (2, [1.0, 2.0, 3.0, 4.0])
    assert ev.eval("b + 1") == 7.0
    assert ev.eval("q") == 15.0  # mm
    assert ev.eval("2*eV") == pytest.approx(2e-6)  # MeV
    assert ev.eval("m[2,1] + m[3]") == 6.0  # 1-based, row-major
    assert ev.eval("sqrt(16)^2 - pi") == pytest.approx(16 - math.pi)
    assert ev.eval("30*deg") == pytest.approx(math.pi / 6)
    for bad in ("nosuch + 1", "1 +", "foo(2)", "m[9]"):
        with pytest.raises(reader.GDMLError):
            ev.eval(bad)


def test_reads_a_small_file(tmp_path):
    p = _gdml(
        tmp_path,
        define='<constant name="L" value="200"/><position name="p1" x="L/2" unit="mm"/>',
        solids='<box name="b" x="L" y="L" z="L" lunit="mm"/>',
        volumes='<volume name="V"><materialref ref="Water"/><solidref ref="b"/></volume>',
        physvols='<physvol name="pv"><volumeref ref="V"/><positionref ref="p1"/></physvol>',
    )
    G = reader.read(p)
    assert G.world == "World" and G.solids["b"].params["x"] == 0.2  # metres
    pl = G.volumes["World"].placements[0]
    assert pl.name == "pv" and pl.transform.position == (0.1, 0.0, 0.0) and pl.child == "V"
    assert G.report == []


def test_lengths_convert_exactly_to_metres(tmp_path):
    """a plain literal in a decimal unit is scaled exactly (rounded once); an expression as Geant4 evaluates it"""
    p = _gdml(
        tmp_path,
        define='<constant name="L" value="0.1"/>',
        solids='<box name="b" x="12.5" y="3" z="L*3" lunit="cm"/>',
        volumes='<volume name="V"><materialref ref="Water"/><solidref ref="b"/></volume>',
        physvols='<physvol><volumeref ref="V"/><position name="q" x="0.3" unit="m"/>'
        '<rotation name="r" z="90" unit="deg"/></physvol>',
    )
    G = reader.read(p)
    b = G.solids["b"].params
    assert b["x"] == 0.125 and b["y"] == 0.03 and b["z"] == 0.1 * 3 * 10.0 / 1000.0
    pl = G.volumes["World"].placements[0]
    assert pl.name == "V_PV" and pl.transform.position == (0.3, 0.0, 0.0)  # Geant4's name of an unnamed physvol
    assert pl.transform.rotation == (0.0, 0.0, 90 * math.pi / 180.0)


@pytest.mark.parametrize(
    "what,kw",
    [
        (
            "unknown constant",
            dict(
                solids='<box name="b" x="NOPE" y="1" z="1"/>',
                volumes='<volume name="V"><materialref ref="W"/><solidref ref="b"/></volume>',
            ),
        ),
        (
            "missing positionref",
            dict(
                solids='<box name="b" x="1" y="1" z="1"/>',
                volumes='<volume name="V"><materialref ref="W"/><solidref ref="b"/></volume>',
                physvols='<physvol><volumeref ref="V"/><positionref ref="nope"/></physvol>',
            ),
        ),
        ("loop", dict(define='<loop for="i" from="1" to="3" step="1"/>')),
        (
            "replicavol",
            dict(
                solids='<box name="b" x="1" y="1" z="1"/>',
                volumes='<volume name="V"><materialref ref="W"/><solidref ref="b"/></volume>',
                physvols='<replicavol number="3"><volumeref ref="V"/></replicavol>',
            ),
        ),
        ("physvol file", dict(physvols='<physvol><file name="other.gdml"/></physvol>')),
        ("volume placed before it is defined", dict(physvols='<physvol><volumeref ref="Later"/></physvol>')),
        ("boolean of a missing solid", dict(solids='<union name="u"><first ref="wbox"/><second ref="nope"/></union>')),
    ],
)
def test_fails_closed(tmp_path, what, kw):
    with pytest.raises(reader.GDMLError):
        reader.read(_gdml(tmp_path, **kw))


DEFECTS = """<matrix name="den" coldim="2" values="1.5e-06 2.2305392e-316 3e-06 0.25"/>
<matrix name="dup" coldim="2" values="1e-06 1 2e-06 2"/>
<matrix name="dup" coldim="2" values="1e-06 1 2e-06 2"/>
<matrix name="odd" coldim="2" values="1e-06 1 2e-06 2 3e-06"/>
<matrix name="broken" coldim="1" values="NOPE"/>
<matrix name="gv" coldim="2" values="1e-06 inf 2e-06 299.8"/>"""


def test_normalises_exporter_defects(tmp_path):
    G = reader.read(_gdml(tmp_path, define=DEFECTS))
    np.testing.assert_array_equal(G.matrix("den"), [[1.5e-6, 0.0], [3e-6, 0.25]])
    np.testing.assert_array_equal(G.matrix("dup"), [[1e-6, 1], [2e-6, 2]])
    np.testing.assert_array_equal(G.matrix("odd"), [[1e-6, 1], [2e-6, 2]])
    rep = "\n".join(G.report)
    assert "'den': 1 denormal" in rep and "re-definition of matrix 'dup'" in rep and "'odd': 5 values" in rep
    assert np.isinf(G.matrix("gv")[0, 1]) and "'gv': 1 non-finite" in rep
    with pytest.raises(reader.GDMLError, match="broken"):
        G.matrix("broken")  # an unused broken table does not block the read; its use does


def test_conflicting_redefinition_is_an_error(tmp_path):
    bad = '<matrix name="dup" coldim="1" values="1"/>\n<matrix name="dup" coldim="1" values="2"/>'
    with pytest.raises(reader.GDMLError, match="conflicting"):
        reader.read(_gdml(tmp_path, define=bad))
    with pytest.raises(reader.GDMLError, match="conflicting"):
        reader.read(_gdml(tmp_path, define='<constant name="c" value="1"/><constant name="c" value="2"/>'))


def test_write_normalised_round_trip(tmp_path):
    src = _gdml(tmp_path, define=DEFECTS.replace('<matrix name="broken" coldim="1" values="NOPE"/>', ""))
    dst = str(tmp_path / "clean.gdml")
    rep = reader.write_normalised(src, dst)
    assert any("denormal" in r for r in rep) and any("dup" in r for r in rep) and any("odd" in r for r in rep)
    G = reader.read(dst)
    assert G.report == []  # nothing left to normalise
    np.testing.assert_array_equal(G.matrix("den"), [[1.5e-6, 0.0], [3e-6, 0.25]])
    np.testing.assert_array_equal(G.matrix("odd"), [[1e-6, 1], [2e-6, 2]])
    np.testing.assert_array_equal(G.matrix("gv"), [[1e-6, 1e30], [2e-6, 299.8]])
    assert "xmlns:xsi" in open(dst).read()


def test_g4_strip():
    assert reader.g4_strip("WCBarrel0x564f8ce03230") == "WCBarrel"
    assert reader.g4_strip("plain") == "plain"
