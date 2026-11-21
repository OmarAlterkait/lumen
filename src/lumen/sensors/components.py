"""A sensor's after-tracking components: time spread, delay, time walk, charge, threshold, pulse.

Each component is a frozen dataclass holding only structure (its form, knot positions, table lengths); every number
it uses is a value passed in at call time, a dict ``p`` of the component's fields (``fields()``: name -> unit, shape),
so a value never recompiles and always carries its derivative. Times are in ns, charges in photoelectrons (pe).

Time spread (TTS), per photoelectron:
  ``gaussian(sigma)``; ``emg(sigma, tau, shift)``: a Gaussian convolved with an exponential tail, shifted;
  ``table(pdf)``: a measured histogram on fixed bins, uniform within each bin.
  ``cdf(p, x)`` / ``sf(p, x)``: the distribution of the arrival-to-photoelectron delay. ``support(ranges)`` is the
  static time range the kernel can reach for any value inside its declared ranges.
Delay (``delay``): a fixed-position delay table (a wavelength shifter's emission time), probabilities as values, one
  per cell of its own axes (a delay that depends on features of the deposit: wavelength bands, re-emission count).
Time walk (``time_walk``): a per-hit shift and extra Gaussian spread against the hit's charge, linear in fixed
  charge knots and constant beyond them.
Charge (SPE): ``gauss_exp(sigma, fraction, tau)``: a Gaussian main peak plus an exponential under-amplified part,
  mean 1 pe by construction; ``table(pdf)``: a measured spectrum on fixed bins. ``pmf(p, edges)`` the masses on a
  charge grid, ``moments(p)`` (mean, mean square), ``sample(rng, p, n)``.
Threshold: ``table(accept)`` acceptance against charge knots; ``poly(coefficients, offset, limit)``: acceptance
  min(1, poly(Q + offset)) below ``limit`` and 1 above (WCSim's SKI form), clipped to [0, 1].
Pulse (the single-photoelectron waveform, unit peak): ``bigauss(t_peak, sigma_rise, sigma_fall)``,
  ``exp2(t_start, tau_rise, tau_fall)``, ``table(shape)`` samples on a fixed time step, linear between them.

A value set's record of a component names its ``form`` and gives its structure and its values; ``Form.from_spec``
builds the component from the record's structure keys (``STATIC``), the rest are its values (``fields()``).

Notation: x a time lag (ns), q a charge (pe), F a CDF, S a survival function.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, ndtr

from lumen._core.domains import FRACTION, POSITIVE
from lumen._core.safe import clip01


class Form:
    """A component form: built from a value-set record by ``from_spec``, its structure the record's ``STATIC`` keys
    (each a list of numbers, the form's fields in order); every other key of the record is a value."""  # noqa: D205, D209

    STATIC: ClassVar[tuple] = ()
    RANGES: ClassVar[dict] = {}  # {field: its admissible range} where a field has one (others: any finite value)

    @classmethod
    def from_spec(cls, spec: dict):
        """Return the component of a value-set record (its ``STATIC`` keys as tuples of floats)."""
        return cls(*(tuple(float(x) for x in spec[k]) for k in cls.STATIC))


SPREAD = 8.0  # Gaussian widths a time kernel reaches: below 1e-15 of its mass lies beyond


# ------------------------------------------------------------------------- emission components (exact)
def _tail(x, s, tau):
    """Return tau exp(-x/tau + s^2/2tau^2) Phi(x/s - s/tau), 0 for tau = 0 (formed with log_ndtr, no overflow)."""
    on = tau > 0.0
    tz = jnp.where(on, tau, 1.0)
    e = jnp.exp(-x / tz + 0.5 * (s / tz) ** 2 + log_ndtr(x / s - s / tz))
    return jnp.where(on, tz * e, 0.0)


def component_cdf(x, tau_d, tau_r, sigma):
    """Return the CDF at x of one emission component (density (e^-t/tau_d - e^-t/tau_r)/(tau_d - tau_r), t >= 0;
    tau_r = 0 a pure exponential, tau_d = 0 prompt) convolved with a Gaussian of width sigma (ns).
    """  # noqa: D205
    s = jnp.maximum(sigma, 1e-6)
    phi = ndtr(x / s)
    den = jnp.where(tau_d > 0.0, tau_d - tau_r, 1.0)
    return jnp.where(tau_d > 0.0, phi - (_tail(x, s, tau_d) - _tail(x, s, tau_r)) / den, phi)


def component_sf(x, tau_d, tau_r, sigma):
    """Return 1 - component_cdf without the cancellation (accurate deep in the right tail)."""
    s = jnp.maximum(sigma, 1e-6)
    phic = ndtr(-x / s)
    den = jnp.where(tau_d > 0.0, tau_d - tau_r, 1.0)
    return jnp.where(tau_d > 0.0, phic + (_tail(x, s, tau_d) - _tail(x, s, tau_r)) / den, phic)


def _interp_cdf(x, edges, pdf):
    """Return the CDF at x of a histogram (masses ``pdf`` on fixed ``edges``, uniform within bins), normalised."""
    c = jnp.concatenate([jnp.zeros(1, pdf.dtype), jnp.cumsum(pdf)])
    return jnp.interp(x, jnp.asarray(edges, jnp.float32), c / c[-1])


# ------------------------------------------------------------------------------------------------ time spread
@dataclass(frozen=True)
class GaussianTTS(Form):
    """A Gaussian transit-time spread of width ``sigma`` (ns)."""

    form: ClassVar[str] = "gaussian"

    RANGES: ClassVar[dict] = {"sigma": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("ns", ())}

    def cdf(self, p, x):
        """Return the delay CDF at x."""
        return component_cdf(x, 0.0, 0.0, p["sigma"])

    def sf(self, p, x):
        """Return the delay survival function at x."""
        return component_sf(x, 0.0, 0.0, p["sigma"])

    def widened(self, p, extra):
        """Return the values with an extra Gaussian spread ``extra`` (ns) added in quadrature."""
        return {**p, "sigma": jnp.sqrt(p["sigma"] ** 2 + extra**2)}

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        return 0.0, p["sigma"] ** 2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return rng.normal(0.0, float(p["sigma"]), n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns over the declared ranges."""
        s = SPREAD * ranges["sigma"][1]
        return -s, s

    def min_width(self, ranges):
        """Return the narrowest width (ns) inside the declared ranges, which sets the time-node spacing."""
        return ranges["sigma"][0]


@dataclass(frozen=True)
class EMGTTS(GaussianTTS):
    """A Gaussian (``sigma``) convolved with an exponential tail (``tau``), shifted by ``shift`` (ns)."""

    form: ClassVar[str] = "emg"

    RANGES: ClassVar[dict] = {"sigma": POSITIVE, "tau": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("ns", ()), "tau": ("ns", ()), "shift": ("ns", ())}

    def cdf(self, p, x):
        """Return the delay CDF at x."""
        return component_cdf(x - p["shift"], p["tau"], 0.0, p["sigma"])

    def sf(self, p, x):
        """Return the delay survival function at x."""
        return component_sf(x - p["shift"], p["tau"], 0.0, p["sigma"])

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        return p["shift"] + p["tau"], p["sigma"] ** 2 + p["tau"] ** 2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return float(p["shift"]) + rng.normal(0.0, float(p["sigma"]), n) + rng.exponential(float(p["tau"]), n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns."""
        s = SPREAD * ranges["sigma"][1]
        return ranges["shift"][0] - s, ranges["shift"][1] + s + 2.0 * SPREAD * ranges["tau"][1]


@dataclass(frozen=True)
class TableTTS(Form):
    """A measured time spread: masses ``pdf`` on the fixed bin edges ``edges`` (ns), uniform within bins."""

    edges: tuple
    form: ClassVar[str] = "table"
    STATIC: ClassVar[tuple] = ("edges_ns",)

    RANGES: ClassVar[dict] = {"pdf": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (len(self.edges) - 1,))}

    def cdf(self, p, x):
        """Return the delay CDF at x."""
        return _interp_cdf(x, self.edges, p["pdf"])

    def sf(self, p, x):
        """Return the delay survival function at x."""
        return 1.0 - self.cdf(p, x)

    def widened(self, p, extra):
        """Return the values unchanged: a table's shape is not widened (the time walk's spread is not applied)."""
        return p

    def moments(self, p):
        """Return (mean, variance) in ns, ns^2."""
        e = jnp.asarray(self.edges)
        c, w = 0.5 * (e[1:] + e[:-1]), p["pdf"] / jnp.sum(p["pdf"])
        m = jnp.sum(w * c)
        return m, jnp.sum(w * ((c - m) ** 2 + (e[1:] - e[:-1]) ** 2 / 12.0))

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return _table_sample(rng, self.edges, p["pdf"], n)

    def support(self, ranges):
        """Return the static (lo, hi) lag range in ns: the table's edges."""
        return self.edges[0], self.edges[-1]

    def min_width(self, ranges):
        """Return the narrowest bin width (ns)."""
        return float(np.min(np.diff(self.edges))) * 4.0


def _table_sample(rng, edges, pdf, n):
    pdf = np.asarray(pdf, np.float64)  # never negative (refused at load: value_sets.DEFAULT_RANGES)
    i = rng.choice(len(pdf), size=n, p=pdf / pdf.sum())
    e = np.asarray(edges, np.float64)
    return e[i] + rng.random(n) * (e[i + 1] - e[i])


# the histogram's cubic (four-node Lagrange) split weights w_o(g), o = -1 .. 2, as polynomials in g (highest first),
# and their antiderivatives: a uniform bin is integrated against them exactly
_CUBIC = {-1: np.poly1d([-1.0, 3.0, -2.0, 0.0]) / 6.0, 0: np.poly1d([1.0, -2.0, -1.0, 2.0]) / 2.0,
          1: np.poly1d([-1.0, 1.0, 2.0, 0.0]) / 2.0, 2: np.poly1d([1.0, 0.0, -1.0, 0.0]) / 6.0}  # fmt: skip
_CUBIC_INT = {o: w.integ() for o, w in _CUBIC.items()}


@functools.lru_cache(maxsize=64)
def delay_taps(edges: tuple, dt: float) -> tuple:
    """Return the static taps of a delay table on nodes of spacing dt (ns): (first tap offset, number of taps, rows,
    columns, weights) with taps[k] = sum over (rows == k) of weights * pdf[columns] (numpy, sparse).

    The weight of bin b on tap k is the mass a delay uniform on bin b puts on node off0 + k: (1 / width) times the
    integral over the bin of the histogram's cubic split weight of that node; a zero-width bin is a point mass (the
    split weights at its point). Built once per (edges, dt), with the grid (``SensorValuesModel.grid``).
    """  # noqa: D205
    e = np.asarray(edges, np.float64) / dt
    lo, hi = int(np.floor(e.min())) - 1, int(np.floor(e.max())) + 2
    acc: dict = {}
    for b in range(len(e) - 1):
        xa, xb = e[b], e[b + 1]
        if xb == xa:  # a point mass at xa
            j = int(np.floor(xa))
            for o, wi in _CUBIC.items():
                acc[(j + o - lo, b)] = acc.get((j + o - lo, b), 0.0) + wi(xa - j)
            continue
        for j in range(int(np.floor(xa)), int(np.ceil(xb))):
            g0, g1 = max(xa, j) - j, min(xb, j + 1) - j
            if g1 <= g0:
                continue
            for o, wi in _CUBIC_INT.items():
                acc[(j + o - lo, b)] = acc.get((j + o - lo, b), 0.0) + (wi(g1) - wi(g0)) / (xb - xa)
    keys = sorted(acc)
    rows = np.array([k[0] for k in keys], np.int32)
    cols = np.array([k[1] for k in keys], np.int32)
    return lo, hi - lo + 1, rows, cols, np.array([acc[k] for k in keys], np.float64)


@functools.lru_cache(maxsize=64)
def _device_taps(edges: tuple, dt: float) -> tuple:
    """Return (first tap offset, M) with taps = M @ pdf, M the dense (taps, bins) matrix of ``delay_taps`` on the
    device in float32 (moved once; a constant under a trace)."""  # noqa: D205, D209
    import jax

    off0, ntap, rows, cols, wts = delay_taps(edges, dt)
    m = np.zeros((ntap, len(edges) - 1))
    m[rows, cols] = wts
    with jax.ensure_compile_time_eval():
        return off0, jnp.asarray(m, jnp.float32)


@dataclass(frozen=True)
class DelayTable:
    """A delay table: probabilities ``pdf`` (values) over bins of fixed ``edges`` (ns, >= 0, non-decreasing; a
    zero-width bin is a point mass), uniform within bins; one table per cell of its ``axes`` (features of the deposit,
    ``lumen.readout.grid.Axis``: e.g. wavelength bands or the re-emission count), ``pdf`` of shape (*axes, bins).

    Applied to the time histogram's node masses, each delay cell's slice through its own table: each bin's density
    is integrated exactly against the histogram's own cubic split weights (no sub-sampling of the bin), so the delay
    is as exact as the node grid. Its taps matrix is static (``delay_taps``, built once with the grid and moved to the
    device once); at run time one matrix product and one FFT per cell.
    """  # noqa: D205

    edges: tuple
    axes: tuple = ()
    form: ClassVar[str] = "table"

    RANGES: ClassVar[dict] = {"pdf": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (*(a.size for a in self.axes), len(self.edges) - 1))}

    def taps(self, dt):
        """Return the static taps on nodes of spacing dt (``delay_taps``)."""
        return delay_taps(tuple(float(x) for x in self.edges), float(dt))

    def _weights(self, p):
        return p["pdf"] / jnp.sum(p["pdf"], -1, keepdims=True)

    def apply(self, p, nodes, dt):
        """Return the node masses (S, nt) after the delay of nodes (S, *axes, nt), summed over the delay's cells; mass
        moved past the last node is dropped."""  # noqa: D205, D209
        off0, m = _device_taps(tuple(float(x) for x in self.edges), float(dt))
        w = self._weights(p)  # (*axes, bins)
        taps = m @ w if w.ndim == 1 else jnp.moveaxis(m @ jnp.moveaxis(w, -1, -2), -1, -2)  # (*axes, taps)
        nt = nodes.shape[-1]
        n = 1 << int(np.ceil(np.log2(nt + m.shape[0] + abs(off0))))
        full = jnp.fft.irfft(jnp.fft.rfft(nodes, n) * jnp.fft.rfft(taps, n), n)
        if full.ndim > 2:  # the delay's cells summed
            full = jnp.reshape(full, (nodes.shape[0], -1, n)).sum(1)
        if off0 >= 0:
            return jnp.concatenate([jnp.zeros((full.shape[0], off0), nodes.dtype), full[:, : nt - off0]], -1)
        return full[:, -off0 : -off0 + nt]

    def mean(self, p):
        """Return the mean delay (ns) per cell (*axes)."""
        e = jnp.asarray(self.edges, jnp.float32)
        return jnp.sum(self._weights(p) * 0.5 * (e[1:] + e[:-1]), -1)

    def sample(self, rng, p, n):
        """Return n draws (numpy) of a delay without axes."""
        return _table_sample(rng, self.edges, p["pdf"], n)

    def support(self, ranges):
        """Return the static (lo, hi) delay range in ns."""
        return self.edges[0], self.edges[-1]


@dataclass(frozen=True)
class TimeWalk(Form):
    """A per-hit time shift ``shift`` and extra Gaussian spread ``sigma`` (ns) at fixed charge knots ``knots`` (pe)."""

    knots: tuple
    form: ClassVar[str] = "table"
    STATIC: ClassVar[tuple] = ("charge_pe",)

    RANGES: ClassVar[dict] = {"sigma": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        k = (len(self.knots),)
        return {"shift": ("ns", k), "sigma": ("ns", k)}

    def at(self, p, q):
        """Return (shift, sigma) at hit charges q (pe), constant beyond the end knots."""
        k = jnp.asarray(self.knots, jnp.float32)
        return jnp.interp(q, k, p["shift"]), jnp.interp(q, k, p["sigma"])


# ------------------------------------------------------------------------------------------------ charge (SPE)
@dataclass(frozen=True)
class GaussExpCharge(Form):
    """A Gaussian main peak (width ``sigma``, pe) plus an exponential part (``fraction``, mean ``tau``), mean 1 pe."""

    form: ClassVar[str] = "gauss_exp"

    RANGES: ClassVar[dict] = {"sigma": POSITIVE, "fraction": FRACTION, "tau": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"sigma": ("pe", ()), "fraction": ("", ()), "tau": ("pe", ())}

    def _peak(self, p):
        return (1.0 - p["fraction"] * p["tau"]) / (1.0 - p["fraction"])

    def cdf(self, p, q):
        """Return the charge CDF at q (pe)."""
        w, tau = p["fraction"], p["tau"]
        expo = jnp.where(q > 0.0, -jnp.expm1(-jnp.maximum(q, 0.0) / jnp.maximum(tau, 1e-6)), 0.0)
        return (1.0 - w) * ndtr((q - self._peak(p)) / p["sigma"]) + w * expo

    def pmf(self, p, edges):
        """Return the masses on charge bins of the given edges (pe)."""
        c = self.cdf(p, jnp.asarray(edges, jnp.float32))
        return c[..., 1:] - c[..., :-1]

    def moments(self, p):
        """Return (mean, mean square) in pe, pe^2."""
        w, tau, m = p["fraction"], p["tau"], self._peak(p)
        return 1.0, (1.0 - w) * (p["sigma"] ** 2 + m**2) + w * 2.0 * tau**2

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        exp = rng.random(n) < float(p["fraction"])
        g = rng.normal(float(self._peak(p)), float(p["sigma"]), n)
        return np.where(exp, rng.exponential(float(p["tau"]), n), g)


@dataclass(frozen=True)
class TableCharge(Form):
    """A measured single-photoelectron spectrum: masses ``pdf`` on fixed bin ``edges`` (pe), uniform within bins."""

    edges: tuple
    form: ClassVar[str] = "table"
    STATIC: ClassVar[tuple] = ("edges_pe",)

    RANGES: ClassVar[dict] = {"pdf": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"pdf": ("", (len(self.edges) - 1,))}

    def cdf(self, p, q):
        """Return the charge CDF at q (pe)."""
        return _interp_cdf(q, self.edges, p["pdf"])

    def pmf(self, p, edges):
        """Return the masses on charge bins of the given edges (pe)."""
        c = self.cdf(p, jnp.asarray(edges, jnp.float32))
        return c[..., 1:] - c[..., :-1]

    def moments(self, p):
        """Return (mean, mean square) in pe, pe^2."""
        e = jnp.asarray(self.edges, jnp.float32)
        w = p["pdf"] / jnp.sum(p["pdf"])
        lo, hi = e[:-1], e[1:]
        return jnp.sum(w * 0.5 * (lo + hi)), jnp.sum(w * (lo * lo + lo * hi + hi * hi) / 3.0)

    def sample(self, rng, p, n):
        """Return n draws (numpy)."""
        return _table_sample(rng, self.edges, p["pdf"], n)


# ------------------------------------------------------------------------------------------------ threshold
@dataclass(frozen=True)
class TableThreshold(Form):
    """Acceptance ``accept`` against fixed charge knots ``knots`` (pe), linear, constant beyond the end knots."""

    knots: tuple
    form: ClassVar[str] = "table"
    STATIC: ClassVar[tuple] = ("charge_pe",)

    RANGES: ClassVar[dict] = {"accept": FRACTION}

    def fields(self):
        """Return the field units and shapes."""
        return {"accept": ("", (len(self.knots),))}

    def accept(self, p, q):
        """Return the acceptance at integrated charges q (pe)."""
        # the knots' values lie in [0, 1] (refused otherwise at load, value_sets.DEFAULT_RANGES), so the interpolant
        # does too; clip01 keeps the derivative of a value on a bound (a plateau at 1) whole
        return clip01(jnp.interp(q, jnp.asarray(self.knots, jnp.float32), p["accept"]))


@dataclass(frozen=True)
class PolyThreshold(Form):
    """Acceptance min(1, sum_k c_k x^k), x = Q + ``offset``, for x below ``limit``; 1 above; clipped to [0, 1]."""

    order: int
    form: ClassVar[str] = "poly"

    @classmethod
    def from_spec(cls, spec: dict):
        """Return the threshold of a record: its order from its coefficients' count."""
        return cls(len(spec["coefficients"]) - 1)

    def fields(self):
        """Return the field units and shapes."""
        return {"coefficients": ("", (self.order + 1,)), "offset": ("pe", ()), "limit": ("pe", ())}

    def accept(self, p, q):
        """Return the acceptance at integrated charges q (pe)."""
        x = q + p["offset"]
        c = p["coefficients"]
        poly = c[-1] * jnp.ones_like(x)
        for k in range(self.order - 1, -1, -1):
            poly = c[k] + x * poly
        return jnp.where(x < p["limit"], jnp.clip(poly, 0.0, 1.0), 1.0)


# ------------------------------------------------------------------------------------------------ pulse
@dataclass(frozen=True)
class BiGaussPulse(Form):
    """Two half-Gaussians about ``t_peak`` (ns): ``sigma_rise`` before, ``sigma_fall`` after; unit peak."""

    form: ClassVar[str] = "bigauss"

    RANGES: ClassVar[dict] = {"sigma_rise": POSITIVE, "sigma_fall": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"t_peak": ("ns", ()), "sigma_rise": ("ns", ()), "sigma_fall": ("ns", ())}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        d = t - p["t_peak"]
        s = jnp.where(d < 0.0, p["sigma_rise"], p["sigma_fall"])
        return jnp.exp(-0.5 * (d / s) ** 2)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return ranges["t_peak"][0] - SPREAD * ranges["sigma_rise"][1], ranges["t_peak"][1] + SPREAD * ranges[
            "sigma_fall"][1]  # fmt: skip


@dataclass(frozen=True)
class Exp2Pulse(Form):
    """A rising and falling exponential from ``t_start`` (ns): e^{-s/tau_fall} - e^{-s/tau_rise}; unit peak."""

    form: ClassVar[str] = "exp2"

    RANGES: ClassVar[dict] = {"tau_rise": POSITIVE, "tau_fall": POSITIVE}

    def fields(self):
        """Return the field units and shapes."""
        return {"t_start": ("ns", ()), "tau_rise": ("ns", ()), "tau_fall": ("ns", ())}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        r, f = p["tau_rise"], p["tau_fall"]
        s = jnp.maximum(t - p["t_start"], 0.0)
        speak = r * f / (f - r) * jnp.log(f / r)
        peak = jnp.exp(-speak / f) - jnp.exp(-speak / r)
        return jnp.where(t > p["t_start"], (jnp.exp(-s / f) - jnp.exp(-s / r)) / peak, 0.0)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return ranges["t_start"][0], ranges["t_start"][1] + 2.0 * SPREAD * ranges["tau_fall"][1]


@dataclass(frozen=True)
class TablePulse(Form):
    """A measured pulse ``shape`` sampled at ``t_first + k step`` (ns, fixed), linear between samples, 0 outside."""

    t_first: float
    step: float
    n: int
    form: ClassVar[str] = "table"
    STATIC: ClassVar[tuple] = ("t_first_ns", "step_ns")

    @classmethod
    def from_spec(cls, spec: dict):
        """Return the pulse of a record: its first sample time, step (ns) and its shape's length."""
        return cls(float(spec["t_first_ns"]), float(spec["step_ns"]), len(spec["shape"]))

    def fields(self):
        """Return the field units and shapes."""
        return {"shape": ("", (self.n,))}

    def shape(self, p, t):
        """Return the pulse at times t (ns) after the photoelectron."""
        k = self.t_first + self.step * jnp.arange(self.n, dtype=jnp.float32)
        return jnp.interp(t, k, p["shape"], left=0.0, right=0.0)

    def support(self, ranges):
        """Return the static (lo, hi) time range in ns."""
        return self.t_first, self.t_first + self.step * (self.n - 1)


TTS_FORMS = {"gaussian": GaussianTTS, "emg": EMGTTS, "table": TableTTS}
CHARGE_FORMS = {"gauss_exp": GaussExpCharge, "table": TableCharge}
THRESHOLD_FORMS = {"table": TableThreshold, "poly": PolyThreshold}
PULSE_FORMS = {"bigauss": BiGaussPulse, "exp2": Exp2Pulse, "table": TablePulse}
