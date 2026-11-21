"""An LAr sensor in LAr: its own class; a same-material active volume is no boundary. (LArSoft's tables and its
reflector, written in its Geant4 form, are lumen-tpc's.)"""

from lumen.build import lookup as regions
from lumen.build.tree import PV, Tree
from lumen.media import bulk, laws
from lumen.surfaces import types


def test_lar_sensor_in_lar_is_its_own_class_and_the_active_volume_is_invisible():
    lar_m = bulk.Medium("lar", laws.constant("index"), laws.constant("length"))
    library = regions.Library(media={"lar": regions.MediumEntry(lar_m, {"index": {"value": 1.36},
                                                                        "absorption": {"value": 1e9}}),
                                     "steel": regions.MediumEntry(bulk.Medium("steel", None), {})},
                              sensors={"arapuca": regions.SensorEntry(types.COUNTING, {"eff": 1.0})})  # fmt: skip
    pvs = (PV("Hall", "HallLV", "Steel"), PV("Cryo", "CryoLV", "LAr", "Hall"), PV("det0", "DetLV", "LAr", "Cryo", 0),
           PV("TPC", "TPCLV", "LAr", "Cryo"))  # fmt: skip
    b = regions.Bindings(
        materials={"LAr": "lar", "Steel": "steel"},
        sensors=(regions.SensorBinding("arapuca", 1, lv="DetLV"),),
    )
    o = regions.compile(library, Tree(pvs), b)
    c = o.class_of
    assert c["TPC"] == c["Cryo"] != c["det0"] and o.pair("Cryo", "TPC") is None
    assert o.pair("Cryo", "det0").surface_type is types.COUNTING and o.sensor_index["det0"] == 0
