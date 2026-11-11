"""The readout after tracking: photoelectrons, waveforms, hits, simulated events and their likelihoods.

    pe = lumen.readout.pe(light, params)                     # expected photoelectrons per sensor
    wf = lumen.readout.expected_waveform(light, params)      # mean and variance per sensor and sample
    eh = lumen.readout.expected_hits(light, params)          # hit probability, charge distribution, first-hit terms
    nll = lumen.readout.hits_nll(eh, lumen.readout.Hits.from_list(sensor, t, q, n_sensors))
    ev = lumen.readout.sample(light, params, seed=1)         # one simulated event (not differentiable)
    q = lumen.readout.time_nodes(light, params)[0]           # expected photoelectrons on the time nodes

A response of your own is one function of the histograms, its parameters declared as keys (``lumen.response``,
``lumen.readout.custom``).

Every function is plain JAX of the histograms and the sensor values, so derivatives in every sensor value are exact
(after-tracking derivatives) and changing a value never recompiles.
"""

from lumen.readout.custom import Response, response, time_nodes
from lumen.readout.detection import pe, pe_error
from lumen.readout.events import SampledEvent, electronics, photoelectrons, sample
from lumen.readout.grid import (
    Grid,
    Histograms,
    charge_cells,
    check_memory,
    from_photons,
    make_grid,
    photon_chunks,
    time_cells,
)
from lumen.readout.hits import ExpectedHits, Hits, expected_hits, hits_nll
from lumen.readout.waveform import PEDensity, Waveform, expected_waveform, pe_density, waveform_nll

__all__ = [
    "ExpectedHits",
    "Grid",
    "Hits",
    "Histograms",
    "PEDensity",
    "Response",
    "SampledEvent",
    "Waveform",
    "charge_cells",
    "check_memory",
    "electronics",
    "from_photons",
    "expected_hits",
    "expected_waveform",
    "hits_nll",
    "make_grid",
    "pe",
    "pe_density",
    "pe_error",
    "photoelectrons",
    "photon_chunks",
    "response",
    "sample",
    "time_cells",
    "time_nodes",
    "waveform_nll",
]
