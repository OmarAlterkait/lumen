"""The public detectors: recipes for the WCSim detectors (SK, WCTE, ANNIE) and the LArSoft ones (SBND, MicroBooNE).

Each is registered in ``lumen.plugins.detectors`` with the dataset ids of its sources under ``LUMEN_DATA``.
"""

from lumen import plugins

_DONE = []


def register_builtins():
    """Register the public detectors (once)."""
    if _DONE:
        return
    _DONE.append(True)
    from lumen.detectors import larsoft, wcsim

    for name, recipe, gdml, inputs in (
        ("sk", wcsim.SK, "gdml/sk.gdml", {}),
        ("wcte", wcsim.WCTE, "gdml/wcte.gdml", {}),
        ("annie", wcsim.ANNIE, "gdml/annie_v2.gdml",
         {"wcsim_source": "wcsim/annie_e88314b", "wcsim_tuning": "wcsim/annie_tuning_parameters.mac"}),
        ("sbnd", larsoft.SBND, "gdml/sbnd_v02_00_full_simulation.gdml", {"pds_mapping": "gdml/sbnd_pds_mapping.json"}),
        ("microboone", larsoft.MICROBOONE, "gdml/microboone.gdml", {}),
    ):  # fmt: skip
        plugins.detectors.register(name, plugins.DetectorPackage(name, recipe, gdml, inputs))
