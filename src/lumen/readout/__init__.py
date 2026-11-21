"""The readout after tracking: photoelectrons, waveforms, hits, simulated events and their likelihoods.

    pe = lumen.readout.pe(light, params)                     # expected photoelectrons per sensor
    wf = lumen.readout.expected_waveform(light, params)      # mean and variance per sensor and sample
    eh = lumen.readout.expected_hits(light, params)          # hit probability, windows' recording probability
    nll = lumen.readout.hits_nll(eh, lumen.readout.Hits.from_list(sensor, t, q, n_sensors))
    ev = lumen.readout.sample(light, params, seed=1)         # one simulated event (not differentiable)
    hits = lumen.simulate(det, src, p, samples=n, seed=0, truth=lumen.readout.Truth()).truth  # one per photon detected
    q = lumen.readout.pe_density(light, params, dark=False)  # expected photoelectron time profile (delay, TTS)

A response of your own is one function of the histograms, its parameters declared as keys (``lumen.response``,
``lumen.readout.custom``).

Every function is plain JAX of the histograms and the sensor values, so derivatives in every sensor value are exact
(after-tracking derivatives) and changing a value never recompiles.
"""

from lumen.readout.custom import Response, response
from lumen.readout.detection import pe, pe_error
from lumen.readout.events import SampledEvent, electronics, sample
from lumen.readout.grid import Grid, Histograms  # the binned deposit records (their cells: lumen.readout.grid)
from lumen.readout.hits import ExpectedHits, Hits, expected_hits, hits_nll
from lumen.readout.truth import Truth, TruthHits
from lumen.readout.waveform import PEDensity, Waveform, expected_waveform, pe_density, waveform_nll

__all__ = [
    "ExpectedHits",
    "Grid",
    "Hits",
    "Histograms",
    "PEDensity",
    "Response",
    "SampledEvent",
    "Truth",
    "TruthHits",
    "Waveform",
    "electronics",
    "expected_hits",
    "expected_waveform",
    "hits_nll",
    "pe",
    "pe_density",
    "pe_error",
    "response",
    "sample",
    "waveform_nll",
]
