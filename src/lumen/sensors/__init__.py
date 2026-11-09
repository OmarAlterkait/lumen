"""Sensor models after tracking: ``lumen:pmt`` (QE, CE, time spread, SPE charge, threshold, pulse, dark rate, front
end) and its shipped value sets (``lumen:<set>@<version>``); see ``lumen.sensors.pmt``.
"""  # noqa: D205

from lumen import plugins
from lumen.sensors.pmt import PMTModel, SensorValuesError, load_model, model_from_dict, write_defaults

plugins.sensor_models.register("lumen:pmt", load_model)

__all__ = ["PMTModel", "SensorValuesError", "load_model", "model_from_dict", "write_defaults"]
